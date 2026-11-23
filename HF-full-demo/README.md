---
title: PentaGemma
emoji: 5️⃣
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: apache-2.0
short_description: Gemma 4 E2B IT but obsessed with 5 sentence summaries
models:
    - google/gemma-4-E2B-it
    - ianro04/PentaGemma
---

A chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).