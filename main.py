import os, io, re
from typing import List, Tuple, Optional

import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

from fastapi import FastAPI, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk.errors import SlackApiError

# 追加：表抽出用
import pdfplumber
import time

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
あなたは専門の研究助手である。アップした要約を重複を避けて統合し、以下の7つの観点で簡潔にまとめよ．
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
CHAT_TEMPLATE_CHUNK = r"""<instruction>
あなたは専門の研究助手である。アップした論文の一部の内容を詳しくまとめよ．
</instruction>
<rule>
・必ず問題設定を記述すること
・論文に書かれていないことは記述しない。
・箇条書きで可読性を重視すること
</rule>
"""
# ========= URL→PDF 解決 =========
ARXIV_RE = re.compile(r"https?://arxiv\.org/(abs|pdf)/(?P<id>[\d\.]+)(?:\.pdf)?")
# 追加：URL抽出ユーティリティ
import re
URL_RE = re.compile(r"https?://\S+")

def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")

def get_urls_from_thread(channel: str, thread_ts: str) -> list[str]:
    """スレッドの親メッセージ＋返信全体からURLを回収"""
    urls = []
    try:
        res = api.conversations_replies(channel=channel, ts=thread_ts, limit=100, inclusive=True)
        for msg in res.get("messages", []):
            urls.extend(extract_urls(msg.get("text", "")))
    except Exception as e:
        print("conversations_replies error:", e)
    # 重複除去
    dedup, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup

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
# ========= 表抽出（画像としてレンダリング） =========
def extract_table_images_from_pdf_bytes(
    raw: bytes,
    dpi: int = 200,
    max_tables: int = 8,
    min_bbox_area: float = 20_000.0
) -> List[Tuple[str, bytes]]:
    """
    pdfplumberで表のbboxを検出 → PyMuPDFでその領域を高解像度レンダリング → PNG化
    戻り値: [(filename, png_bytes), ...]
    """
    results: List[Tuple[str, bytes]] = []
    try:
        # pdfplumber: テーブル検出用
        pdfp = pdfplumber.open(io.BytesIO(raw))
        # PyMuPDF: レンダリング用
        doc = fitz.open(stream=raw, filetype="pdf")

        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)

        for pidx, page in enumerate(pdfp.pages, start=1):
            try:
                tbls = page.find_tables()  # Tableオブジェクトのリスト（bboxあり）
            except Exception:
                tbls = []

            if not tbls:
                continue

            fitz_page = doc[pidx - 1]
            for tidx, t in enumerate(tbls, start=1):
                if not hasattr(t, "bbox") or t.bbox is None:
                    continue
                x0, top, x1, bottom = t.bbox
                # bboxの面積でノイズ除去
                if (x1 - x0) * (bottom - top) < min_bbox_area:
                    continue

                rect = fitz.Rect(x0, top, x1, bottom)
                try:
                    pix = fitz_page.get_pixmap(matrix=mat, clip=rect, alpha=False)
                    png = pix.tobytes("png")
                    fname = f"tableimg_p{pidx}_{tidx}.png"
                    results.append((fname, png))
                    if len(results) >= max_tables:
                        pdfp.close()
                        doc.close()
                        return results
                except Exception as e:
                    # 個別表の描画失敗はスキップ
                    print("render table bbox error:", e)

        pdfp.close()
        doc.close()
    except Exception as e:
        print("extract_table_images_from_pdf_bytes error:", e)

    return results

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
def upload_table_images_as_replies(channel: str, thread_ts: str, tables: List[Tuple[str, bytes]], limit: int = 6):
    for fname, data in tables[:limit]:
        try:
            api.files_upload_v2(
                channel=channel,
                file=io.BytesIO(data),
                filename=fname,
                title=fname,
                thread_ts=thread_ts,
                #initial_comment="検出された表（画像、自動抽出）"
            )
        except SlackApiError as e:
            print("files_upload_v2 (table images) error:", e)
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
    chunk_size = 5000  # 文字ベース
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)] or [full_text]

    partials: List[str] = []
    for ch in chunks:
        prompt = CHAT_TEMPLATE_CHUNK + "\n<chunk>\n" + ch+"\n</chunk>\n"
        time.sleep(1)  # API制限回避
        try:
            resp = client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            partials.append(resp.choices[0].message.content.strip())
        except Exception as e:
            msg = str(e)
            if _is_token_limit_error_message(msg):
                return None, f"Groqのトークン（コンテキスト長）上限に達したため、要約できませんでした。管理者に本文量を減らすかチャンクサイズを小さくするよう要請してください．{msg}"
            # その他のAPIエラー
            return None, f"Groq APIエラーが発生しました: {msg}"

    if len(partials) == 1:
        return partials[0], None

    # reduce
    time.sleep(1)  # API制限回避
    reduce_prompt = CHAT_TEMPLATE + "\n" + "<summarize_chunks>" + "\n\n".join(partials)+"\n"+"</summarize_chunk>"+"\n"
    try:
        resp = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
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
def get_urls_from_thread(channel: str, thread_ts: str) -> list[str]:
    """スレッドの親メッセージ→スレッド全体からURLを回収"""
    urls = []
    try:
        res = api.conversations_replies(channel=channel, ts=thread_ts, limit=50, inclusive=True)
        for msg in res.get("messages", []):
            urls.extend(extract_urls(msg.get("text", "")))
    except Exception as e:
        print("conversations_replies error:", e)
    # 重複除去を軽く
    seen = set()
    dedup = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup
"""
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
        
        api.chat_postMessage(
            channel=ch,
            text=f"リンクを受け付けました。処理中です...\n(処理が終わるまで数分かかります)",
            thread_ts=ts
        )

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
        api.chat_postMessage(
            channel=ch,
            text=f"図の抽出を開始します...",
            thread_ts=ts
        )
        try:
            figs = extract_figures_from_pdf_bytes(raw_pdf, min_area=200_000)
            if ch and ts and figs:
                upload_figures_as_replies(ch, ts, figs, limit=6)
        except Exception as e:
            logger.exception(e)
            post_error_message(ch, ts, f"図の抽出またはアップロードに失敗しました: {e}")
 		# 4) 表抽出（画像）→ スレッドへアップ
        #try:
        #    table_imgs = extract_table_images_from_pdf_bytes(raw_pdf, dpi=220, max_tables=8, min_bbox_area=20_000.0)
        #    if ch and ts and table_imgs:
        #        upload_table_images_as_replies(ch, ts, table_imgs, limit=6)
        #except Exception as e:
        #    logger.exception(e)
        #    post_error_message(ch, ts, f"表（画像）の抽出またはアップロードに失敗しました: {e}")
        api.chat_postMessage(
            channel=ch,
            text=f"要約が完了しました。",
            thread_ts=ts
        )
"""
# ========= Slack: app_mention（スレッドだけ反応） =========
@app.event("app_mention")
def handle_app_mention(body, event, say, logger):
    channel = event.get("channel")
    thread_ts = event.get("thread_ts")  # スレッド内メッセージのみ反応
    if not channel or not thread_ts:
        # スレッド外でメンションされても反応しない
        return

    # 1) メンション本文 → 2) スレッド全体 の順でURL探索
    text = event.get("text", "")
    urls = extract_urls(text)
    if not urls:
        urls = get_urls_from_thread(channel, thread_ts)

    # 許可ドメインに限定
    urls = [u for u in urls if is_allowed(u)]
    if not urls:
        try:
            api.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=":information_source: スレッドに処理対象の論文URLが見つかりません。\n"
                     "arXiv / CVF / IEEE / ACM などのURLを貼ってから `@sumxiv` とメンションしてください。"
            )
        except Exception as e:
            logger.exception(e)
        return

    # 複数ある場合は先頭のみ処理（必要ならループに拡張可）
    url = urls[0]
    pdf_url = resolve_pdf_url(url)
    if not pdf_url:
        post_error_message(channel, thread_ts, "PDFリンクを見つけられませんでした。HTMLページの場合はPDF直リンクをお願いします。")
        return

    raw_pdf = download_pdf(pdf_url)
    if not raw_pdf:
        post_error_message(channel, thread_ts, "PDFのダウンロードに失敗しました。サイズやアクセス制限をご確認ください。")
        return

    # 進捗メッセ
    try:
        api.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="処理を開始しました。要約を作成中…"
        )
    except Exception as e:
        logger.exception(e)

    # 1) テキスト抽出 → 要約（Groq）
    text = extract_text_from_pdf_bytes(raw_pdf)
    summary, err = summarize_text_with_groq(text)
    if err:
        post_error_message(channel, thread_ts, err)
        return

    # 2) 要約をスレッドに投稿（アンフールは使わない運用）
    try:
        api.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=f"*要約（@sumxiv 指示で実行）*\n{summary}\n\n<{url}|元のリンク>"
        )
    except Exception as e:
        logger.exception(e)
        post_error_message(channel, thread_ts, f"要約の投稿に失敗しました: {e}")
        return

    # 3) 図抽出 → スレッドへアップ
    try:
        figs = extract_figures_from_pdf_bytes(raw_pdf, min_area=200_000)
        if figs:
            upload_figures_as_replies(channel, thread_ts, figs, limit=6)
    except Exception as e:
        logger.exception(e)
        post_error_message(channel, thread_ts, f"図の抽出/アップロードに失敗しました: {e}")

    # 4) 表（画像）抽出 → スレッドへアップ
    try:
        table_imgs = extract_table_images_from_pdf_bytes(raw_pdf, dpi=220, max_tables=8, min_bbox_area=20_000.0)
        if table_imgs:
            upload_table_images_as_replies(channel, thread_ts, table_imgs, limit=6)
    except Exception as e:
        logger.exception(e)
        post_error_message(channel, thread_ts, f"表（画像）の抽出/アップロードに失敗しました: {e}")

    # 完了メッセ
    try:
        api.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="完了しました ✅"
        )
    except Exception as e:
        logger.exception(e)
