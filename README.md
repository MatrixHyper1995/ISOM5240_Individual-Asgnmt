# 🖼️ Image Storyteller

上传一张图片 → AI 用 Hugging Face Transformers pipeline 看图提取细节、生成一段 **50–100 词的英文故事** → 朗读出来。

ISOM5240 深度学习的商业应用 · Individual Assignment

## 技术栈

- **读图提取细节**：`pipeline("image-to-text")` → `Salesforce/blip-image-captioning-base`
- **故事生成**：`pipeline("text-generation")` → `distilgpt2`
- **可选 LLM 方式**：HF Inference API 远程调用 VLM（侧边栏可切换）
- **朗读**：edge-tts（微软 Edge 在线 TTS，免费无 key）
- **框架**：Streamlit（部署于 Streamlit Community Cloud）

## 目录结构

```
ISOM5240_IA_Jingxiang MA/
├── app.py                     # 主程序
├── requirements.txt           # 依赖
├── .streamlit/
│   ├── config.toml            # 上传上限 + 主题色
│   └── secrets.example.toml   # HF Token 模板（仅 LLM 模式需要）
├── .gitignore
├── 设计文档.md
└── README.md
```

## 本地运行（可选）

```bash
pip install -r requirements.txt
# 若启用 LLM 模式：cp .streamlit/secrets.example.toml .streamlit/secrets.toml 并填入 token
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. 把本目录推到一个 GitHub 仓库（`app.py` 在仓库根目录）。
2. 打开 <https://share.streamlit.io> → New app → 选仓库 / 分支 / 入口 `app.py` → Deploy。
3. （可选，仅 LLM 模式）在 app 的 **Settings → Secrets** 填入 `HF_TOKEN = "***"`。

## 使用说明

- 上传图片（PNG / JPG / JPEG / BMP / TIFF，≤ 12 MB）。
- 侧边栏选择风格、长度、音色、语速。
- 点 **✨ Generate Story** 生成故事，随后自动朗读。
- 生成方式默认 **Pipeline**（本地小模型，课程要求）；可切 **LLM API**（需 HF Token）。
- 改音色/语速只重新合成语音，不会重新生成故事。

## 常见报错

| 提示 | 含义 |
|------|------|
| Invalid or unauthorized HF Token | LLM 模式的 Token 无效 |
| This model is gated | 需在 Hugging Face 网页接受该模型的许可 |
| Rate limited / too large | 免费额度不足，需 Inference PRO |
| Please upload an image first | 尚未上传图片 |
| File is too large | 超过 12 MB |
