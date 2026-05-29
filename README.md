---
title: Receipt Intelligence
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🧾 Receipt Intelligence

AI-powered receipt scanning and spending analytics, built for Kenyan retail receipts.

## Features

- **OCR + Document AI** — PaddleOCR extracts text; LayoutLMv3 classifies fields (store, items, total, KRA PIN, cashier, date)
- **Date extraction** — pulls transaction dates from raw OCR text
- **Smart categorisation** — keyword rules → sentence embeddings → optional Qwen2.5 LLM
- **Store name normalisation** — maps OCR noise to clean store names
- **Multi-user auth** — Supabase Auth with username/password; per-user data isolation
- **Batch upload** — process multiple images at once or upload a ZIP file
- **Camera capture** — take photos directly in the browser
- **AI insights** — Qwen2.5-1.5B generates specific, data-grounded spending insights
- **Time analytics** — spend over time, day-of-week patterns, trip gaps, cumulative curve
- **CSV exports** — all items, all receipts, or per-receipt downloads

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| OCR | PaddleOCR 2.7.3 |
| Document AI | LayoutLMv3 (fine-tuned) |
| LLM | Qwen2.5-1.5B-Instruct (optional) |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Database | Supabase (PostgreSQL) |
| Auth | Supabase Auth |

## ⚠️ Free Tier Notice

This Space runs on CPU. Qwen2.5 LLM features (AI insights, LLM categorisation) may be slow or unavailable. All other features work normally.

## Setup (for contributors)

1. Fork this repo
2. Set up a Supabase project and run `schema.sql`
3. Add `SUPABASE_URL`, `SUPABASE_KEY`, and `HF_MODEL_REPO` to Space secrets
4. Add your fine-tuned LayoutLMv3 model to a private HF Hub repo

## License

MIT
