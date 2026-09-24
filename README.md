# 🖼️ Image Storyteller · 图片讲故事

> ISOM5240 深度学习的商业应用 · Individual Assignment
> ISOM5240 Deep Learning for Business Applications · Individual Assignment

上传一张图片 → AI 用 Hugging Face Transformers pipeline 读图提取细节、生成一段 **50–100 词的英文故事** → 朗读出来。

Upload an image → AI extracts details with a Hugging Face Transformers pipeline, writes a **50–100 word English story**, and reads it aloud.

---

## 技术栈 · Tech Stack

| 环节 / Stage | 方案 / Solution |
|---|---|
| 读图（提取细节）<br>Image understanding | `pipeline("image-text-to-text")` → `microsoft/Florence-2-base`（固定 / fixed） |
| 故事生成<br>Story generation | `pipeline("text-generation")` → `SupraLabs/StorySupra-10M`（默认 / default）、`distilgpt2`；或 LLM API（`GLM-OCR` / `Qwen2.5-VL-7B`） |
| 朗读 / TTS | edge-tts（微软 Edge 在线 TTS，免费无 key / free, no key） |
| 框架 / Framework | Streamlit（部署于 Streamlit Community Cloud） |

> 说明：读图环节固定用 Florence-2（出详细 caption）；侧边栏的「Story model」只切换**编故事**模型。
> Note: Image understanding always uses Florence-2 (detailed caption); the sidebar "Story model" only switches the **story-writing** model.

## 目录结构 · Project Structure

```
ISOM5240_IA_Jingxiang MA/
├── app.py                     # 主程序 / Main app
├── requirements.txt           # 依赖 / Dependencies
├── .streamlit/
│   ├── config.toml            # 上传上限 + 主题色 / Upload limit + theme
│   └── secrets.example.toml   # HF Token 模板（仅 LLM 模式）/ Token template
├── .gitignore
├── 设计文档.md                # 技术设计文档 / Design doc
└── README.md
```

## 本地运行 · Run Locally

```bash
pip install -r requirements.txt
# 若启用 LLM 模式，先配置 token：
# To use LLM mode, configure the token first:
cp .streamlit/secrets.example.toml .streamlit/secrets.toml   # 填入 HF_TOKEN / fill in HF_TOKEN
streamlit run app.py
```

## 部署到 Streamlit Cloud · Deploy to Streamlit Cloud

1. 把本目录推到一个 GitHub 仓库（`app.py` 在仓库根目录）。
   Push this folder to a GitHub repo (`app.py` at the repo root).
2. 打开 <https://share.streamlit.io> → New app → 选仓库 / 分支 / 入口 `app.py` → Deploy。
   Open <https://share.streamlit.io> → New app → pick repo / branch / entry `app.py` → Deploy.
3. （可选，仅 LLM 模式）在 **Settings → Secrets** 填 `HF_TOKEN = "***"`。
   (Optional, LLM mode only) Fill `HF_TOKEN = "***"` in **Settings → Secrets**.

## 使用说明 · How to Use

1. 上传图片（PNG / JPG / JPEG / BMP / TIFF，≤ 12 MB）。
   Upload an image (PNG / JPG / JPEG / BMP / TIFF, ≤ 12 MB).
2. 侧边栏选择故事模型、风格、长度、音色、语速。
   Pick the story model, style, length, voice and speed in the sidebar.
3. 点 **✨ Generate Story** 生成故事，随后自动朗读。
   Click **✨ Generate Story** to generate, then it reads aloud automatically.
4. 改音色/语速只重新合成语音，不会重新生成故事。
   Changing voice/speed only re-synthesizes audio, without regenerating the story.

## 常见报错 · Common Errors

| 提示 / Message | 含义 / Meaning |
|---|---|
| Invalid or unauthorized HF Token | LLM 模式的 Token 无效 / Invalid token for LLM mode |
| This model is gated | 需在 HF 网页接受该模型许可 / Accept the model license on HF |
| Rate limited / too large | 免费额度不足，需 Inference PRO / Needs Inference PRO |
| Please upload an image first | 尚未上传图片 / No image uploaded yet |
| File is too large | 超过 12 MB / Exceeds 12 MB |
