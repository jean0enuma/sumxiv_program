import os, io, re
from typing import List, Tuple, Optional

import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

from fastapi import FastAPI, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk.errors import SlackApiError

# ========= 環境変数 =========
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ========= Slack Bolt =========
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
api = app.client

# ========= FastAPI =========
fastapi_app = FastAPI()
handler = SlackRequestHandler(app)

@fastapi_app.get("/")
async def root():
    return {"status": "ok"}

@fastapi_app.get("/healthz")
async def healthz():
    return {"ok": True}

@fastapi_app.post("/slack/events")
async def slack_events(req: Request):
    return await handler.handle(req)

# ========= 要約テンプレート =========
CHAT_TEMPLATE = r"""<instruction>
あなたは専門の研究助手である。アップロードされた論文もしくはURL先の論文を以下のフォーマットに従って詳しくまとめよ。
</instruction>
<format>
1. どんなもの？
2. 問題設定は?
3. 先行研究と比べてどこがすごい？
4. 技術や手法のキモはどこ？
5. どうやって有効だと検証した？
6. 議論はある？
7. 次に読むべき論文は?
</format>
<rule>
・必ず問題設定を記述すること
・論文に書かれていないことは記述しない。
・箇条書きで可読性を重視すること
</rule>
"""

# ========= URL→PDF 解決 =========
ARXIV_RE = re.compile(r"https?://arxiv\.org/(abs|pdf)/(?P<id>[\d\.]+)(?:\.pdf)?")

def resolve_pdf_url(url: str) -> Optional[str]:
    m = ARXIV_RE.match(url)
    if m:
        arxiv_id = m.group("id")
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if url.lower().endswith(".pdf"):
        return url
    try:
        html = requests.get(url, timeout=20).text
    except Exception:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = (a.get_text() or "").lower()
        if href.lower().endswith(".pdf") or "pdf" in href.lower() or "pdf" in text:
            return requests.compat.urljoin(url, href)
    return None

def download_pdf(pdf_url: str) -> Optional[bytes]:
    try:
        r = requests.get(pdf_url, timeout=60)
        r.raise_for_status()
        if int(r.headers.get("Content-Length", "0")) > 30_000_000:
            return None
        return r.content
    except Exception:
        return None

# ========= PDF→テキスト =========
def extract_text_from_pdf_bytes(raw: bytes, max_chars: int = 120_000) -> str:
    parts = []
    with fitz.open(stream=raw, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text())
            if sum(len(x) for x in parts) > max_chars:
                break
    return "\n".join(parts)

# ========= 図抽出（埋め込み画像） =========
def extract_figures_from_pdf_bytes(raw: bytes, min_area: int = 200_000) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    with fitz.open(stream=raw, filetype="pdf") as doc:
        for pidx, page in enumerate(doc, start=1):
            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                try:
                    if pix.width * pix.height >= min_area:
                        if pix.n >= 4 or pix.colorspace is None:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        out.append((f"fig_p{pidx}_{xref}.png", pix.tobytes("png")))
                finally:
                    pix = None
    return out

# ========= Slack 投稿ユーティリティ =========
def post_unfurl_summary(event: dict, url: str, summary_text: str):
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*要約（自動生成）*\n{summary_text}"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": f"_URL: <{url}>_"}]}
    ]
    args = {"unfurls": {url: {"blocks": blocks}}}
    if "unfurl_id" in event and "source" in event:
        args.update({"unfurl_id": event["unfurl_id"], "source": event["source"]})
    else:
        if event.get("channel"):
            args.update({"channel": event["channel"]})
        if event.get("message_ts") or event.get("ts"):
            args.update({"ts": event.get("message_ts") or event.get("ts")})
    api.chat_unfurl(**args)

def upload_figures_as_replies(channel: str, thread_ts: str, figures: List[Tuple[str, bytes]], limit: int = 5):
    for fname, data in figures[:limit]:
        try:
            api.files_upload_v2(
                channel=channel,
                file=io.BytesIO(data),
                filename=fname,
                title=fname,
                thread_ts=thread_ts,
                initial_comment="検出された図（自動抽出）"
            )
        except SlackApiError as e:
            print("files_upload_v2 error:", e)

def post_error_message(channel: Optional[str], thread_ts: Optional[str], text: str):
    """エラーをSlackに人間可読で通知（スレッドがあればスレッドに返信）"""
    if not channel:
        return
    try:
        api.chat_postMessage(
            channel=channel,
            text=f":warning: {text}",
            thread_ts=thread_ts
        )
    except SlackApiError as e:
        print("chat_postMessage error:", e)

# ========= Groq で要約（エラー通知対応） =========
def _is_token_limit_error_message(msg: str) -> bool:
    """Groqのトークン上限に起因しそうなメッセージを簡易判定"""
    msg_l = msg.lower()
    patterns = [
        "rate limit",      
        "RateLimitError",              
    ]
    return any(p in msg_l for p in patterns)

def summarize_text_with_groq(full_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    return (summary, error_message)
    error_message が None でなければ呼び出し側でSlack通知する
    """
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY が未設定のため要約できません。"

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    # チャンク→マージ（最小実装）
    chunk_size = 6000  # 文字ベース
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)] or [full_text]

    partials: List[str] = []
    for ch in chunks:
        prompt = CHAT_TEMPLATE + "\n\n本文抜粋:\n" + ch
        try:
            resp = client.chat.completions.create(
                model="groq/compound-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            partials.append(resp.choices[0].message.content.strip())
        except Exception as e:
            msg = str(e)
            if _is_token_limit_error_message(msg):
                return None, "Groqのトークン（コンテキスト長）上限に達したため、要約できませんでした。本文量を減らすかチャンクサイズを小さくしてください。"
            # その他のAPIエラー
            return None, f"Groq APIエラーが発生しました: {msg}"

    if len(partials) == 1:
        return partials[0], None

    # reduce
    reduce_prompt = CHAT_TEMPLATE + "\n\n以下のチャンク要約を重複を避けて統合せよ:\n" + "\n\n".join(partials)
    try:
        resp = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[{"role": "user", "content": reduce_prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip(), None
    except Exception as e:
        msg = str(e)
        if _is_token_limit_error_message(msg):
            return None, "Groqのトークン（コンテキスト長）上限に達したため、要約の統合に失敗しました。"
        return None, f"Groq APIエラーが発生しました（reduce段階）: {msg}"

# ========= 許可ドメイン =========
ALLOW_HOSTS = (
    "arxiv.org",
    "openaccess.thecvf.com",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
    "proceedings.mlr.press",
)

def is_allowed(url: str) -> bool:
    return any(h in url for h in ALLOW_HOSTS)

# ========= Slack: link_shared =========
@app.event("link_shared")
def handle_link_shared_events(body, event, logger, say):
    links = event.get("links", [])
    if not links:
        return
    for link in links:
        url = link.get("url")
        if not url or not is_allowed(url):
            continue

        pdf_url = resolve_pdf_url(url)
        if not pdf_url:
            continue

        raw_pdf = download_pdf(pdf_url)
        if not raw_pdf:
            continue

        # チャンネル/スレッド識別（エラー通知に使う）
        ch = event.get("channel")
        ts = event.get("message_ts") or event.get("ts")

        # 1) テキスト抽出 → 要約（Groq, エラー捕捉）
        text = extract_text_from_pdf_bytes(raw_pdf)
        summary, err = summarize_text_with_groq(text)

        if err:
            post_error_message(ch, ts, err)
            # エラー時はアンフールを行わず次へ
            continue

        # 2) アンフールで要約
        try:
            post_unfurl_summary(event, url, summary or "")
        except Exception as e:
            logger.exception(e)
            post_error_message(ch, ts, f"アンフール中にエラーが発生しました: {e}")

        # 3) 図抽出 → スレッドへアップ
        try:
            figs = extract_figures_from_pdf_bytes(raw_pdf, min_area=200_000)
            if ch and ts and figs:
                upload_figures_as_replies(ch, ts, figs, limit=6)
        except Exception as e:
            logger.exception(e)
            post_error_message(ch, ts, f"図の抽出またはアップロードに失敗しました: {e}")
