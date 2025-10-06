import os, io, re
from typing import List, Tuple, Optional
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk.errors import SlackApiError
import pdfplumber
import time
from PIL import Image # 画像のリサイズ用に追加
from openai import OpenAI
#import os
#os.kill(os.getpid(), 9)
# ========= 環境変数 =========
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

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
# 最終的な要約を生成するためのテンプレート
CHAT_TEMPLATE = r"""<instruction>
あなたは専門の研究助手である。以下に示される論文の各ページの要約を統合し、重複を避けながら、下記の7つの観点で簡潔な最終要約を作成せよ。
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
・必ず問題設定を記述すること。
・論文に書かれていないことは記述しないこと。
・各項目は箇条書きで、可読性を重視すること。
</rule>
<example>
*1. どんなもの？*
   - 連続手話認識のための新しい学習手法であるSelf-Mutual Knowledge Distillation (SMKD)を提案。
   - 視覚モジュールと文脈モジュールの両方の識別能力を同時に強化することを目的とする。
   - 視覚モジュールは空間的・短期的時間情報に、文脈モジュールは長期的時間情報に着目するよう学習。
*2.問題設定は?*
   -連続手話認識では、視覚モジュールが空間的・短期的時間情報を、文脈モジュールが長期的時間情報を捉える必要がある。
   -しかし、従来のend-to-endの学習では、視覚モジュールが最適な特徴を学習することが難しい。
*3. 先行研究と比べてどこがすごい？*
   - 従来のend-to-endの学習では、視覚モジュールが最適な特徴を学習することが難しいという問題に対処。
   - SMKDにより、視覚モジュールの識別力を高め、文脈モジュールが視覚情報により注目するようにしている。
   - これにより、両モジュールの協調を維持しつつ、視覚モジュールの性能を引き出している。

*4. 技術や手法のキモはどこ？*
   - 視覚モジュールと文脈モジュールに同じ分類器の重みを共有させ、CTCロスで同時に学習。
   - グロスセグメンテーションを導入し、CTCロスによるスパイク現象に対処、視覚モジュールの飽和を減らす。
   - 学習の最終段階で重み共有を解消し、文脈モジュールの制約を緩和。
*5. どうやって有効だと検証した？*
   - PHOENIX14とPHOENIX14-Tの2つの連続手話認識データセットで実験。
   - RGB画像のみを使用した手法の中で最先端の性能を達成（PHOENIX14: Dev 20.8%, Test 21.0%, PHOENIX14-T: Dev 20.8%, Test 22.4%）。
   - 重み共有による視覚・文脈モジュールの協調、グロスセグメンテーションの効果を複数の実験で確認。

*6. 議論はある？*
   - 文脈情報も認識タスクにとって重要なので、最終的な学習段階では2つのモジュールの重み共有を解消すべきだと議論。
   - これにより、文脈モジュールがより長期的な時間情報に特化できるようになる。

*7. 次に読むべき論文は？*
   - CNNとLSTMを用いた連続手話認識の研究 [15,23]
   - 教師なし学習を用いた連続手話認識の手法 [6,35]
   - これらの関連研究と比較することで、SMKDの有効性をより詳細に分析できる。
</example>
"""

# 各ページ（画像とテキスト）を個別に要約するためのテンプレート
CHAT_TEMPLATE_PAGE_VISION = r"""<instruction>
あなたは専門の研究助手である。以下に示す論文の1ページ分の画像と、抽出されたテキストを読み、そのページの内容をできるだけ詳しく要約せよ。特に、画像の内容を考慮に入れて要約すること。
</instruction>

<rule>
・画像とテキストの両方から得られる情報を統合して要約を作成すること。
・テキストにない情報で、画像から明確に読み取れる内容は含めること。
・論文に書かれていないことは記述しないこと。
・箇条書きで、可読性を重視すること。
</rule>
"""
# 結合画像を用いた要約テンプレート
CHAT_TEMPLATE_ONE_VISION = r"""<instruction>
あなたは専門の研究助手である。以下に示す論文の画像と、抽出されたテキストを読み、そのページの内容を詳細かつ簡潔に要約せよ。
</instruction>

<rule>
・画像とテキストの両方から得られる情報を統合して要約を作成すること。
・テキストにない情報で、画像から明確に読み取れる内容は含めること。
・論文に書かれていないことは記述しないこと。
・箇条書きで、可読性を重視すること。
</rule>
<example>
1. どんなもの？
   - 連続手話認識のための新しい学習手法であるSelf-Mutual Knowledge Distillation (SMKD)を提案。
   - 視覚モジュールと文脈モジュールの両方の識別能力を同時に強化することを目的とする。
   - 視覚モジュールは空間的・短期的時間情報に、文脈モジュールは長期的時間情報に着目するよう学習。
2.問題設定は?
   -連続手話認識では、視覚モジュールが空間的・短期的時間情報を、文脈モジュールが長期的時間情報を捉える必要がある。
   -しかし、従来のend-to-endの学習では、視覚モジュールが最適な特徴を学習することが難しい。
3. 先行研究と比べてどこがすごい？
   - 従来のend-to-endの学習では、視覚モジュールが最適な特徴を学習することが難しいという問題に対処。
   - SMKDにより、視覚モジュールの識別力を高め、文脈モジュールが視覚情報により注目するようにしている。
   - これにより、両モジュールの協調を維持しつつ、視覚モジュールの性能を引き出している。

4. 技術や手法のキモはどこ？
   - 視覚モジュールと文脈モジュールに同じ分類器の重みを共有させ、CTCロスで同時に学習。
   - グロスセグメンテーションを導入し、CTCロスによるスパイク現象に対処、視覚モジュールの飽和を減らす。
   - 学習の最終段階で重み共有を解消し、文脈モジュールの制約を緩和。
5. どうやって有効だと検証した？
   - PHOENIX14とPHOENIX14-Tの2つの連続手話認識データセットで実験。
   - RGB画像のみを使用した手法の中で最先端の性能を達成（PHOENIX14: Dev 20.8%, Test 21.0%, PHOENIX14-T: Dev 20.8%, Test 22.4%）。
   - 重み共有による視覚・文脈モジュールの協調、グロスセグメンテーションの効果を複数の実験で確認。

6. 議論はある？
   - 文脈情報も認識タスクにとって重要なので、最終的な学習段階では2つのモジュールの重み共有を解消すべきだと議論。
   - これにより、文脈モジュールがより長期的な時間情報に特化できるようになる。

7. 次に読むべき論文は？
   - CNNとLSTMを用いた連続手話認識の研究 [15,23]
   - 教師なし学習を用いた連続手話認識の手法 [6,35]
   - これらの関連研究と比較することで、SMKDの有効性をより詳細に分析できる。
</example>
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
        if int(r.headers.get("Content-Length", "0")) > 50_000_000: # 50MB に上限緩和
            return None
        return r.content
    except Exception:
        return None

# ========= PDFページを画像として抽出 & テキスト抽出 =========
def extract_pages_data_from_pdf_bytes(raw: bytes, dpi: int = 200) -> List[Tuple[bytes, str]]:
    """PDFのバイトデータから、ページごとの画像（PNGバイト）とテキストを抽出する"""
    pages_data = []
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                
                # ページ画像をPNGとしてレンダリング
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
                img_bytes = pix.tobytes("png")
                
                # ページテキストを抽出
                text = page.get_text()
                
                pages_data.append((img_bytes, text))
    except Exception as e:
        print(f"Error extracting pages data from PDF: {e}")
        return []
    return pages_data

# ========= openrouter で要約（Vision対応） =========
def _is_token_limit_error_message(msg: str) -> bool:
    msg_l = msg.lower()
    patterns = ["rate limit", "ratelimiterror", "context_length_exceeded", "too many tokens"]
    return any(p in msg_l for p in patterns)
    

def summarize_pages_with_openrouter_vision(pages_data: List[Tuple[bytes, str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    ページごとの画像とテキストをVision APIで要約し、最後に全体を統合する。
    戻り値: (最終要約, エラーメッセージ)
    """
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY が未設定のため要約できません。"

    client = OpenAI(
  		base_url="https://openrouter.ai/api/v1",
  		api_key=OPENROUTER_API_KEY,
	)
    page_summaries: List[str] = []

    # 1. 各ページをVision APIで要約
    for i, (img_bytes, page_text) in enumerate(pages_data, start=1):
        messages_content = []
        messages_content.append({"type": "text", "text": f"論文の{i}/{len(pages_data)}ページ目です。"})
        if i==1:#タイトルを抽出
            resp = client.chat.completions.create(
                model="mgoogle/gemma-3-27b-it:free",  # Openrouter Visionモデル
				messages=[{"role": "user", "content": [{"type": "text", "text": "この論文のタイトルを教えてください。"},{"type": "image_url","image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"}}]}],
				temperature=0.01,
				#max_tokens=1024 # ページ要約の出力トークン数制限
			)
            title= resp.choices[0].message.content.strip()
        # 画像を追加
        messages_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"}
        })
        
        # 抽出したテキストを追加 (もしあれば)
        if page_text.strip():
            messages_content.append({"type": "text", "text": f"このページから抽出されたテキスト:\n```\n{page_text}\n```"})
        
        # プロンプトを追加
        messages_content.append({"type": "text", "text": CHAT_TEMPLATE_PAGE_VISION})

        #time.sleep(2)  # APIレート制限を避ける

        try:
            resp = client.chat.completions.create(
                model="google/gemma-3-27b-it:free",  # Openrouter Visionモデル
                messages=[{"role": "user", "content": messages_content}],
                temperature=0.01,
                #max_tokens=1024 # ページ要約の出力トークン数制限
            )
            summary = resp.choices[0].message.content.strip()
            page_summaries.append(f"## Page {i} Summary:\n{summary}")
        except Exception as e:
            try:
                resp = client.chat.completions.create(
					model="mistralai/mistral-small-3.2-24b-instruct:free",  # Openrouter Visionモデル
					messages=[{"role": "user", "content": messages_content}],
					temperature=0.01,
					#max_tokens=1024 # ページ要約の出力トークン数制限
				)
                summary = resp.choices[0].message.content.strip()
                page_summaries.append(f"## Page {i} Summary:\n{summary}")
            except Exception as e:
                msg = str(e)
                if _is_token_limit_error_message(msg):
                    return None, None, f"Openrouter Vision APIのトークン上限のため、{i}ページ目の要約に失敗しました。{msg}"
                return None, None, f"Openrouter Vision APIエラーが発生しました ({i}ページ目): {msg}"

    if not page_summaries:
        return None,None,None, "PDFから画像またはテキストが抽出できなかったか、全てのページが処理できませんでした。"
    time.sleep(3)
    # 2. 全ページの要約を統合
    #time.sleep(1)  # APIレート制限を避ける
    combined_summaries = "\n\n".join(page_summaries)
    reduce_prompt = f"{CHAT_TEMPLATE}\n<page_summaries>\n{combined_summaries}\n</page_summaries>"
	
    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",  # 統合には高性能なテキストモデルを使用
            messages=[{"role": "user", "content": reduce_prompt}],
            temperature=0.1,
        )
        return title,resp.choices[0].message.content.strip(), None
    except Exception as e:
        msg = str(e)
        if _is_token_limit_error_message(msg):
            return None,None, f"Openrouter APIのトークン上限のため、要約の統合に失敗しました。: {msg}"
        return None,None, f"Openrouter APIエラーが発生しました（統合段階）: {msg}"
def summarize_pages_with_openrouter_onevision(pages_data: List[Tuple[bytes, str]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    PDFの全ページ画像を1枚に結合し、Vision APIで一度に要約する。
    戻り値: (タイトル, 最終要約, エラーメッセージ)
    """
    if not OPENROUTER_API_KEY:
        return None, None, "OPENROUTER_API_KEY が未設定のため要約できません。"

    if not pages_data:
        return None, None, "要約対象のページデータがありません。"
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
	#0. 先にタイトルを抽出しておく
    first_img_bytes, first_page_text = pages_data[0]
	
    try:
        resp = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",  # Openrouter Visionモデル
            messages=[{"role": "user", "content": [{"type": "text", "text": "この論文のタイトルを教えてください。"},{"type": "image_url","image_url": {"url": f"data:image/png;base64,{base64.b64encode(first_img_bytes).decode('utf-8')}"}}]}],
			temperature=0.01,
			#max_tokens=1024 # ページ要約の出力トークン数制限
		)
        title= resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        if _is_token_limit_error_message(msg):
            return None, None, f"Openrouter Vision APIのトークン上限のため、タイトルの抽出に失敗しました。{msg}"
        return None, None, f"Openrouter Vision APIエラーが発生しました (タイトル抽出): {msg}"
    # 1. 全ページの画像をPillowを使って1枚の縦長の画像に結合
    try:
        images = [Image.open(io.BytesIO(img_bytes)) for img_bytes, _ in pages_data]
        
        # APIの画像サイズ上限を超える場合、ここでリサイズ処理を挟むことも可能
        # 例: for i, img in enumerate(images): images[i] = img.resize(...)

        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        print(f"結合画像サイズ: {max_width} x {total_height}") # デバッグ用にサイズを出力

        combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))

        current_y = 0
        for img in images:
            combined_image.paste(img, (0, current_y))
            current_y += img.height

        # 結合した画像をバイトデータに変換 (JPEGで圧縮しサイズを削減)
        buffer = io.BytesIO()
        combined_image.save(buffer, format="JPEG", quality=85)
        combined_image_bytes = buffer.getvalue()
        print(f"結合画像のバイトサイズ: {len(combined_image_bytes)} bytes") # デバッグ用にバイトサイズを出力

    except Exception as e:
        return None, None, f"画像の結合処理中にエラーが発生しました: {e}"

    # 2. Vision APIで要約
    try:
        # 結合した画像をBase64エンコード
        base64_image = base64.b64encode(combined_image_bytes).decode('utf-8')

        messages_content = [
            {"type": "text", "text": CHAT_TEMPLATE_ONE_VISION},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
        
        # 大きな画像を扱えるVisionモデルを選択（例: Gemini, Claude 3）
        # ご利用のモデルが大きな画像に対応しているかご確認ください
        resp = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free", 
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.1,
            #max_tokens=4096 # 要約の出力用にトークン数を確保
        )
        
        summary = resp.choices[0].message.content.strip()

        return title, summary, None

    except Exception as e:
        msg = str(e)
        if _is_token_limit_error_message(msg):
            return None, None, f"Openrouter APIのトークン/画像サイズ上限のため要約に失敗しました。ページ数の少ないPDFでお試しください。: {msg}"
        return None, None, f"Openrouter APIエラーが発生しました: {msg}"
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

def upload_files_as_replies(channel: str, thread_ts: str, files: List[Tuple[str, bytes]], initial_comment: str, limit: int = 6):
    for fname, data in files[:limit]:
        try:
            api.files_upload_v2(
                channel=channel,
                file=io.BytesIO(data),
                filename=fname,
                title=fname,
                thread_ts=thread_ts,
                initial_comment=initial_comment
            )
        except SlackApiError as e:
            print(f"files_upload_v2 error for {fname}: {e}")

def post_error_message(channel: Optional[str], thread_ts: Optional[str], text: str):
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

# ========= 許可ドメイン =========
ALLOW_HOSTS = (
    "arxiv.org", "openaccess.thecvf.com", "ieeexplore.ieee.org",
    "dl.acm.org", "link.springer.com", "proceedings.mlr.press",
)

def is_allowed(url: str) -> bool:
    return any(h in url for h in ALLOW_HOSTS)

# ========= Slack: link_shared イベントハンドラ =========
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

        ch = event.get("channel")
        ts = event.get("message_ts") or event.get("ts")
        
        api.chat_postMessage(
            channel=ch,
            text=f"論文リンクを受け付けました。要約を開始します... (ページ数が多いほど時間がかかります)",
            thread_ts=ts
        )

        raw_pdf = download_pdf(pdf_url)
        if not raw_pdf:
            post_error_message(ch, ts, f"PDFのダウンロードに失敗しました。ファイルサイズが大きすぎるか、アクセスできません。: {pdf_url}")
            continue

        # 1. ページごとに画像とテキストを抽出
        # Vision APIは画像サイズに制限があるため、事前にリサイズが必要になる可能性も考慮に入れる
        pages_data = extract_pages_data_from_pdf_bytes(raw_pdf, dpi=300) # Vision API向けにDPIを調整
        if not pages_data:
            post_error_message(ch, ts, "PDFからページ画像またはテキストを抽出できませんでした。")
            continue
        
        # 2. ページごとの画像とテキストをVision APIで要約・統合
        title,summary, err = summarize_pages_with_openrouter_vision(pages_data)
        if err:
            post_error_message(ch, ts, err)
            continue
		#3. アンフールなしでスレッドに要約を投稿
        api.chat_postMessage(channel=ch, text=f"*{title}*\n*論文要約*\n{summary}", thread_ts=ts)
        """
		# 3. アンフールで要約を投稿
        try:
            post_unfurl_summary(event, url, summary or "要約の生成に失敗しました。")
        except Exception as e:
            logger.exception(e)
            post_error_message(ch, ts, f"アンフール中にエラーが発生しました: {e}")
            # アンフール失敗時もスレッドに要約を投稿
            api.chat_postMessage(channel=ch, text=f"*要約（自動生成）*\n{summary}", thread_ts=ts)
        """
        # 4. 図と表を抽出してスレッドにアップロード（既存機能）
        try:
            figs = extract_figures_from_pdf_bytes(raw_pdf, min_area=150_000)
            if ch and ts and figs:
                upload_files_as_replies(ch, ts, figs, initial_comment="検出された図（自動抽出）", limit=6)
            
            #table_imgs = extract_table_images_from_pdf_bytes(raw_pdf, dpi=220, max_tables=8, min_bbox_area=20_000.0)
            #if ch and ts and table_imgs:
            #    upload_files_as_replies(ch, ts, table_imgs, initial_comment="検出された表（画像として抽出）", limit=6)
        except Exception as e:
            logger.exception(e)
            post_error_message(ch, ts, f"図の抽出・アップロード中にエラーが発生しました: {e}")

        time.sleep(20)  # 少し待ってから完了メッセージを投稿
		# リンクを投稿したユーザーIDを取得
        user_id = event.get("user")
        
        # ユーザーIDが取得できればメンションを付けて通知、できなければ通常のメッセージを投稿
        if user_id:
            completion_text = f"<@{user_id}> 要約が完了しました。"
        else:
            completion_text = "要約が完了しました。"
            
        api.chat_postMessage(channel=ch, text=completion_text, thread_ts=ts)
# openrouter Vision APIの利用にはBase64エンコードが必要なため追加
import base64