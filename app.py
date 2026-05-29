import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_call_stack_level"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import tempfile
import zipfile
import io

from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import (
    Image,
    ImageDraw,
    ImageOps,
    ImageEnhance,
)

from paddleocr import PaddleOCR

from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Receipt Intelligence",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── lucide icon helpers ───────────────────────────────────────────────────────

def icon(name: str, size: int = 18) -> str:
    url = f"https://unpkg.com/lucide-static@latest/icons/{name}.svg"
    return (
        f'<img src="{url}" width="{size}" height="{size}" '
        f'style="vertical-align:middle;filter:invert(1);margin-right:6px;" />'
    )

def icon_label(icon_name: str, label: str, size: int = 18) -> str:
    return icon(icon_name, size) + f'<span style="vertical-align:middle">{label}</span>'

def metric_tile(icon_name: str, label: str, value: str) -> str:
    ico = icon(icon_name, size=16)
    return f"""
    <div class="metric-tile">
        <div class="tile-label">{ico}{label}</div>
        <div class="tile-value">{value}</div>
    </div>"""

# ── styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a2f3e; }
[data-testid="stSidebar"] *        { color: #e0e6f0 !important; }

div[data-testid="metric-container"] {
    background: #1c2233; border: 1px solid #2a2f3e;
    border-radius: 12px; padding: 16px;
}

.insight-card {
    background: #1c2233; border-left: 3px solid #4caf96;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
    color: #c8d4e8; font-size: 14px;
    display: flex; align-items: flex-start; gap: 10px;
}
.insight-card.warn { border-left-color: #e8a838; }
.insight-card.info { border-left-color: #5b8dee; }
.insight-card img  { margin-top: 2px; flex-shrink: 0; }

.metric-icon-row {
    display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap;
}
.metric-tile {
    background: #1c2233; border: 1px solid #2a2f3e; border-radius: 12px;
    padding: 18px 22px; flex: 1; min-width: 140px;
    display: flex; flex-direction: column; gap: 6px;
}
.metric-tile .tile-label {
    font-size: 12px; color: #8892a4;
    display: flex; align-items: center; gap: 6px;
}
.metric-tile .tile-value {
    font-size: 22px; font-weight: 700; color: #e8edf5;
}

.auth-card {
    background: #1c2233; border: 1px solid #2a2f3e;
    border-radius: 16px; padding: 36px 40px;
    max-width: 420px; margin: 60px auto 0;
}
.auth-title {
    font-size: 22px; font-weight: 700; color: #e8edf5;
    margin-bottom: 6px; display: flex; align-items: center; gap: 10px;
}
.auth-sub { font-size: 13px; color: #8892a4; margin-bottom: 24px; }

.batch-row {
    background: #1c2233; border: 1px solid #2a2f3e; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 12px; font-size: 13px;
}
.batch-ok   { border-left: 3px solid #4caf96; }
.batch-fail { border-left: 3px solid #e85858; }
.batch-pending { border-left: 3px solid #8892a4; }

.stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"]      { color: #8892a4; border-radius: 8px; }
.stTabs [aria-selected="true"]    { background: #1c2233 !important; color: #e8edf5 !important; }
</style>
""", unsafe_allow_html=True)

# ── credentials ───────────────────────────────────────────────────────────────
# Works in both Colab (userdata) and HF Spaces (environment variables / secrets)

def _get_secret(key: str) -> str | None:
    # 1. Colab Secrets
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return val
    except Exception:
        pass
    # 2. HF Spaces secrets / any environment variable
    return os.environ.get(key)

SUPABASE_URL  = _get_secret("SUPABASE_URL")
SUPABASE_KEY  = _get_secret("SUPABASE_KEY")
HF_MODEL_REPO = _get_secret("HF_MODEL_REPO")   # e.g. "your-hf-username/layoutlmv3-receipts"
HF_TOKEN      = _get_secret("HF_TOKEN")         # HuggingFace read token (for private model repo)

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "Supabase credentials not found. "
        "Add SUPABASE_URL and SUPABASE_KEY to your Space secrets "
        "(Settings → Variables and secrets)."
    )
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── environment detection ─────────────────────────────────────────────────────

IS_HF_SPACE = os.environ.get("SPACE_ID") is not None   # HF sets this automatically
IS_COLAB    = not IS_HF_SPACE and os.path.exists("/content")

# ── paths ─────────────────────────────────────────────────────────────────────

if IS_COLAB:
    # Colab: model on Drive, images on Drive
    MODEL_DIR  = "/content/drive/MyDrive/receipts/layoutlmv3_model"
    UPLOAD_DIR = "/content/drive/MyDrive/receipts/dashboard_uploads"
else:
    # HF Spaces / any server: model from HF Hub, images in local /tmp
    MODEL_DIR  = HF_MODEL_REPO or "microsoft/layoutlmv3-base"  # fallback to base if not set
    UPLOAD_DIR = "/tmp/receipt_uploads"

QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

os.makedirs(UPLOAD_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── free tier detection ───────────────────────────────────────────────────────
# On HF free tier there is no GPU. We keep Qwen available but warn users.

HAS_GPU      = torch.cuda.is_available()
QWEN_WARNING = (
    "⚠️ This Space runs on CPU. Qwen2.5 LLM features may take 2-5 minutes "
    "or fail due to memory limits. Embedding categorisation is recommended."
)

# ── session state defaults ────────────────────────────────────────────────────

for key, default in {
    "user":         None,
    "profile":      None,
    "qwen_loaded":  False,   # track whether Qwen loaded successfully
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── ML loaders ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_layoutlm():
    kwargs = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    processor = LayoutLMv3Processor.from_pretrained(
        MODEL_DIR, apply_ocr=False, use_fast=False, **kwargs
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR, **kwargs)
    model.to(device).eval()
    # label_map: try local file first (Colab), then HF Hub file
    try:
        if IS_COLAB:
            with open(f"{MODEL_DIR}/label_map.json") as f:
                label_map = json.load(f)
        else:
            from huggingface_hub import hf_hub_download
            lm_path = hf_hub_download(
                repo_id=MODEL_DIR, filename="label_map.json", token=HF_TOKEN
            )
            with open(lm_path) as f:
                label_map = json.load(f)
    except Exception as e:
        st.error(f"Could not load label_map.json: {e}")
        st.stop()
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    return processor, model, id2label

processor, layoutlm_model, id2label = load_layoutlm()


@st.cache_resource
def load_ocr():
    """
    PaddleOCR 2.8.0 + PaddlePaddle 3.0.0 compatible loader.
    Uses CPU-safe arguments only to avoid deprecated/incompatible parameters.
    """
    import paddle

    paddle.set_device("cpu")

    return PaddleOCR(
        lang="en",
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
        enable_mkldnn=False,
        cpu_threads=1,
        det_limit_side_len=960,
        det_db_thresh=0.3,
        det_db_box_thresh=0.6,
        det_db_unclip_ratio=1.5,
        max_batch_size=1,
        use_dilation=False,
        rec_batch_num=1,
        cls_batch_num=1,
    )

ocr = load_ocr()


@st.cache_resource
def load_qwen():
    """
    Lazy Qwen loader — returns (tokenizer, model, error_message).
    On CPU/low-memory environments this may fail gracefully.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16 if HAS_GPU else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

# Qwen is loaded lazily on first use, not at startup
# This prevents the Space from crashing on boot if RAM is tight
_qwen_cache = {"loaded": False, "tokenizer": None, "model": None, "error": None}

def get_qwen():
    """Return (tokenizer, model) or (None, None) if unavailable."""
    if not _qwen_cache["loaded"]:
        tok, mdl, err = load_qwen()
        _qwen_cache.update({"loaded": True, "tokenizer": tok, "model": mdl, "error": err})
        if err:
            print(f"Qwen load failed: {err}")
    return _qwen_cache["tokenizer"], _qwen_cache["model"]


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

embedding_model = load_embedding_model()

# ── constants ─────────────────────────────────────────────────────────────────

MAX_SEQ_LENGTH = 512

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1c2233",
    plot_bgcolor="#1c2233",
    font=dict(color="#c8d4e8"),
)

CATEGORY_COLORS = {
    "electronics":        "#4caf96",
    "mobile_accessories": "#38d4d4",
    "beverages":          "#5b8dee",
    "dairy":              "#38d4d4",
    "bakery":             "#e87838",
    "meat":               "#e85858",
    "household":          "#e8a838",
    "toiletries":         "#c07be8",
    "tobacco":            "#6a5acd",
    "groceries":          "#20b2aa",
    "other":              "#8892a4",
}

VALID_CATEGORIES = list(CATEGORY_COLORS.keys())

# ── store name normalisation ──────────────────────────────────────────────────

STORE_NAME_MAP = [
    ("naivas",         "Naivas"),
    ("carrefour",      "Carrefour"),
    ("majid",          "Carrefour"),
    ("quickmart",      "Quickmart"),
    ("chandarana",     "Chandarana"),
    ("foodplus",       "Chandarana"),
    ("foocpus",        "Chandarana"),
    ("cleanshelf",     "Cleanshelf"),
    ("tuskys",         "Tuskys"),
    ("uchumi",         "Uchumi"),
    ("shoprite",       "Shoprite"),
    ("eastmatt",       "Eastmatt"),
    ("dungahii",       "Dungahii Camp"),
    ("china square",   "China Square"),
    ("healthcare farm","Healthcare Farm"),
    ("lc waikiki",     "LC Waikiki"),
    ("bata",           "Bata"),
    ("kfc",            "KFC"),
    ("kentucky",       "KFC"),
    ("java house",     "Java House"),
    ("java",           "Java House"),
    ("chicken inn",    "Chicken Inn"),
    ("artcaffe",       "Artcaffe"),
    ("subway",         "Subway"),
    ("pizza inn",      "Pizza Inn"),
    ("galitos",        "Galitos"),
    ("steers",         "Steers"),
    ("shell",          "Shell"),
    ("total",          "TotalEnergies"),
    ("kenol",          "Kenol"),
    ("rubis",          "Rubis"),
]

_NOISE = re.compile(
    r'\b(ltd|limited|hypermarkets?|supermarkets?|superstore|stores?|'
    r'shops?|branch|kenya|nairobi|kisumu|mombasa|westend|ground\s*floor|'
    r'wing\s*[a-z]|united\s*mall|kikapuk\w*|bonge|aas|narvas|'
    r'p/?o\s*box|\d{4,})\b',
    flags=re.IGNORECASE,
)

def normalise_store_name(raw: str) -> str:
    if not raw:
        return raw
    lower = raw.lower().strip()
    for pattern, canonical in STORE_NAME_MAP:
        if pattern in lower:
            return canonical
    cleaned = _NOISE.sub(' ', lower).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.title() if cleaned else raw.title()

# ── date extraction ───────────────────────────────────────────────────────────

def extract_dates_from_text(full_text: str) -> str | None:
    """
    Extract the most likely transaction date from raw OCR text.
    Returns a YYYY-MM-DD string or None.
    """
    date_patterns = [
        r"(\d{4}[-/.]\d{2}[-/.]\d{2})",
        r"(\d{2}[-/.]\d{2}[-/.]\d{4})",
        r"(\d{2}\s*[A-Za-z]{3,9}\s*\d{4})",
        r"([A-Za-z]{3,9}\s*\d{2},?\s*\d{4})",
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})",
        r"(\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4})",
        r"([A-Za-z]{3,9}[-/]\d{1,2}[-/]\d{2,4})",
        r"(\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4})",
        r"(\d{4}\s*[-/]\s*\d{2}\s*[-/]\s*\d{2})",
        r"(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\s+\d{1,2}:\d{2})",
        r"(\d{4}[-/.]\d{2}[-/.]\d{2}\s+\d{1,2}:\d{2}:\d{2})",
        r"(?:DATE|DT|D/T)[:\-\s]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})",
        r"(?:DATE|DT|D/T)[:\-\s]*(\d{4}[-/.]\d{2}[-/.]\d{2})",
        r"(?:DATE\s*TIME)[:\-\s]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})",
        r"(\d{2}\s+[A-Za-z]{3,9}\s+\d{4})",
        r"([A-Za-z]{3,9}\s+\d{2},?\s+\d{4})",
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2,4}\b',
    ]
    found = []
    for pattern in date_patterns:
        for match in re.findall(pattern, full_text, re.IGNORECASE):
            cleaned = clean_and_validate_date(match)
            if cleaned:
                found.append(cleaned)
    return found[0] if found else None


def clean_and_validate_date(date_string: str) -> str | None:
    if not date_string:
        return None
    cleaned = date_string.strip()
    month_map = {
        'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06',
        'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12',
        'january':'01','february':'02','march':'03','april':'04','june':'06',
        'july':'07','august':'08','september':'09','october':'10',
        'november':'11','december':'12',
    }
    manual_formats = [
        "%Y-%m-%d","%Y/%m/%d","%Y.%m.%d",
        "%d-%m-%Y","%d/%m/%Y","%d.%m.%Y",
        "%m-%d-%Y","%m/%d/%Y","%m.%d.%Y",
        "%d %b %Y","%d %B %Y","%b %d %Y","%B %d %Y",
        "%d-%b-%Y","%d-%B-%Y","%b-%d-%Y","%B-%d-%Y",
        "%d/%b/%Y","%d/%B/%Y","%b/%d/%Y","%B/%d/%Y",
        "%d.%b.%Y","%d.%B.%Y","%b.%d.%Y","%B.%d.%Y",
        "%d-%m-%y","%d/%m/%y","%d.%m.%y",
        "%m-%d-%y","%m/%d/%y","%m.%d.%y",
    ]
    for fmt in manual_formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            if 2020 <= parsed.year <= datetime.now().year + 1:
                return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        parts = re.split(r'[-/.\s]+', cleaned)
        if len(parts) == 3:
            day, month, year = parts
            month_lower = month.lower()
            month_num = month_map.get(month_lower, month.zfill(2))
            day_num   = day.zfill(2)
            year_num  = f"20{year}" if len(year) == 2 and int(year) <= 50 else \
                        f"19{year}" if len(year) == 2 else year
            if (1 <= int(day_num) <= 31 and 1 <= int(month_num) <= 12 and
                    2020 <= int(year_num) <= datetime.now().year + 1):
                reconstructed = f"{year_num}-{month_num}-{day_num}"
                final = datetime.strptime(reconstructed, "%Y-%m-%d")
                return final.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None

# ── categorisation ────────────────────────────────────────────────────────────

KEYWORD_RULES = {
    "electronics":        ["earphone","headphone","charger","usb","cable",
                           "speaker","adapter","power bank","flash disk",
                           "oraimo","neckband","wireless","microphone",
                           "collar clip","outdoor mic"],
    "mobile_accessories": ["airtime","bundle","simcard","sim card",
                           "screen protector","phone cover","data bundle"],
    "household":          ["soap","detergent","bleach","tissue","omo",
                           "jik","cleaning spray","packaging bag",
                           "non wooven","nonwoven","latex ring"],
    "tobacco":            ["cigarette","marlboro","sportsman","embassy",
                           "lighter","matchbox"],
    "beverages":          ["coke","sprite","juice","water","tea","coffee",
                           "jinro","soju","stoney","tangawizi","cocktail",
                           "maccoffee","cappuccino","mint","can"],
    "dairy":              ["milk","yogurt","cheese","butter","brookside"],
    "bakery":             ["bread","cake","bun","mandazi","cookie"],
    "meat":               ["beef","chicken","sausage","fish","wings"],
    "toiletries":         ["toothpaste","shampoo","lotion","vaseline","deodorant"],
    "groceries":          ["rice","basmati","flour","sugar","salt",
                           "carrot","tomato","onion","potato","vegetable"],
}

CATEGORY_DESCRIPTIONS = {
    "electronics":        ["earphones","charger","usb cable","power bank",
                           "speaker","wireless neckband","microphone clip"],
    "mobile_accessories": ["sim card","airtime","phone cover","screen protector"],
    "household":          ["soap","detergent","bleach","tissue","omo","jik","packaging bag"],
    "tobacco":            ["cigarettes","lighter","matchbox"],
    "beverages":          ["coke","juice","water","tea","coffee","soju",
                           "cocktail can","cappuccino sachet"],
    "dairy":              ["milk","yogurt","cheese","butter"],
    "bakery":             ["bread","cake","bun","mandazi"],
    "meat":               ["beef","chicken","sausages","fish","chicken wings"],
    "toiletries":         ["toothpaste","shampoo","lotion","vaseline"],
    "groceries":          ["basmati rice","flour","sugar","carrots","tomatoes","onions"],
}

@st.cache_resource
def build_category_embeddings():
    return {
        cat: embedding_model.encode(descs, convert_to_numpy=True, normalize_embeddings=True)
        for cat, descs in CATEGORY_DESCRIPTIONS.items()
    }

CATEGORY_EMBEDDINGS = build_category_embeddings()


def fast_categorise_item(item_name: str) -> str:
    if not item_name:
        return "other"
    try:
        lower = str(item_name).lower().strip()
        for category, keywords in KEYWORD_RULES.items():
            for kw in keywords:
                if kw in lower:
                    return category
        emb = embedding_model.encode(lower, convert_to_numpy=True, normalize_embeddings=True)
        best, score = "other", -1
        for cat, cat_emb in CATEGORY_EMBEDDINGS.items():
            s = cosine_similarity([emb], [cat_emb])[0][0]
            if s > score:
                score, best = s, cat
        return best if score >= 0.45 else "other"
    except Exception as e:
        print(f"Fast categorisation error: {e}")
        return "other"

# ── Qwen helpers ──────────────────────────────────────────────────────────────

def qwen_available() -> bool:
    """Check if Qwen is loaded and ready."""
    tok, mdl = get_qwen()
    return tok is not None and mdl is not None


def qwen_chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str | None:
    """
    Run Qwen inference. Returns None if Qwen is unavailable.
    Uses lazy loader so the app doesn't crash on startup if RAM is tight.
    """
    tokenizer, model = get_qwen()
    if tokenizer is None or model is None:
        return None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def qwen_categorise_item(item_name: str) -> str:
    if not item_name:
        return fast_categorise_item(item_name)
    system = (
        "You are a retail item categoriser. "
        "Given an item name, return ONLY one category word from this list: "
        + ", ".join(VALID_CATEGORIES)
        + ". No explanation, no punctuation, just the single category word."
    )
    try:
        reply = qwen_chat(system, f"Item: {item_name}", max_new_tokens=10)
        if reply is None:
            return fast_categorise_item(item_name)
        reply = reply.lower().strip()
        for word in reply.split():
            clean = re.sub(r'[^a-z_]', '', word)
            if clean in VALID_CATEGORIES:
                return clean
        return fast_categorise_item(item_name)
    except Exception as e:
        print(f"Qwen categorisation error: {e}")
        return fast_categorise_item(item_name)


def categorize_item(item_name: str, use_llm: bool = False) -> str:
    if use_llm and not qwen_available():
        return fast_categorise_item(item_name)
    return qwen_categorise_item(item_name) if use_llm else fast_categorise_item(item_name)


def build_insights_context(receipts_df: pd.DataFrame, items_df: pd.DataFrame) -> str:
    lines = []
    total_spend = receipts_df["receipt_total"].sum()
    n           = len(receipts_df)
    avg_b       = receipts_df["receipt_total"].mean()
    max_b       = receipts_df["receipt_total"].max()
    positive    = receipts_df[receipts_df["receipt_total"] > 0]["receipt_total"]
    min_b       = positive.min() if not positive.empty else 0
    std_b       = receipts_df["receipt_total"].std()

    lines.append(f"RECEIPTS: {n} total")
    lines.append(f"TOTAL SPEND: KES {total_spend:,.0f}")
    lines.append(f"BASKET SIZE: avg KES {avg_b:,.0f}, min KES {min_b:,.0f}, "
                 f"max KES {max_b:,.0f}, std KES {std_b:,.0f}")

    store_visits = receipts_df["store_name"].value_counts()
    store_spend  = receipts_df.groupby("store_name")["receipt_total"].sum().sort_values(ascending=False)
    store_avg    = receipts_df.groupby("store_name")["receipt_total"].mean()
    lines.append("STORE VISITS: " + ", ".join(f"{s}={v}" for s, v in store_visits.items()))
    lines.append("STORE TOTAL SPEND: " + ", ".join(
        f"{s}=KES {v:,.0f}" for s, v in store_spend.items()))
    lines.append("STORE AVG BASKET: " + ", ".join(
        f"{s}=KES {store_avg[s]:,.0f}" for s in store_spend.index))

    if not items_df.empty:
        freq  = items_df["item_name"].value_counts()
        top10 = freq.head(10)
        lines.append("TOP 10 ITEMS BY FREQUENCY: " + ", ".join(
            f'"{i}" x{c}' for i, c in top10.items()))
        once = (freq == 1).sum()
        lines.append(f"UNIQUE ONE-OFF ITEMS: {once} out of {len(freq)} distinct items")
        staples = freq[freq >= 2].index.tolist()
        if staples:
            lines.append("REPEAT ITEMS (2+ times): " + ", ".join(
                f'"{s}" x{freq[s]}' for s in staples[:8]))
        cat_spend = items_df.groupby("category")["line_total"].sum().sort_values(ascending=False)
        cat_pct   = (cat_spend / cat_spend.sum() * 100).round(1)
        lines.append("SPEND BY CATEGORY: " + ", ".join(
            f"{c}=KES {cat_spend[c]:,.0f} ({cat_pct[c]}%)" for c in cat_spend.index))
        top_price = (
            items_df[items_df["unit_price"] > 0]
            .groupby("item_name")["unit_price"].max()
            .sort_values(ascending=False).head(5)
        )
        lines.append("MOST EXPENSIVE ITEMS: " + ", ".join(
            f'"{i}"=KES {p:,.0f}' for i, p in top_price.items()))
        top_cum = (
            items_df.groupby("item_name")["line_total"]
            .sum().sort_values(ascending=False).head(5)
        )
        lines.append("HIGHEST TOTAL SPEND PER ITEM: " + ", ".join(
            f'"{i}"=KES {v:,.0f}' for i, v in top_cum.items()))
        outliers = receipts_df[receipts_df["receipt_total"] > avg_b + 2 * std_b]
        if not outliers.empty:
            lines.append("UNUSUALLY LARGE RECEIPTS: " + ", ".join(
                f'{r["store_name"]} KES {r["receipt_total"]:,.0f}'
                for _, r in outliers.iterrows()))
    return "\n".join(lines)


def qwen_generate_insights(receipts_df: pd.DataFrame, items_df: pd.DataFrame) -> str:
    if receipts_df.empty:
        return ""
    if not qwen_available():
        return ""
    context = build_insights_context(receipts_df, items_df)
    system = """You are a sharp, data-driven personal finance analyst reviewing a customer's shopping receipts from Kenyan stores.

Output exactly 6 insights. Each insight is ONE sentence on its own line, prefixed with exactly one of these tags:
[SPEND] [STORE] [ITEM] [CATEGORY] [HABIT] [ACTION]

Hard rules:
- Every insight MUST quote actual numbers, store names, or item names from the data.
- DO NOT write generic advice. BAD: "consider budgeting." GOOD: "[SPEND] Your KES 6,254 Chandarana receipt is 4x your average basket of KES 1,541."
- [ACTION] must give ONE concrete, specific recommendation — name the store, item, or category.
- [HABIT] must describe a real pattern visible in the data.
- No bullet points, no numbering, no preamble. Just 6 tagged lines.
- Do not invent data. Only use what is provided."""
    try:
        result = qwen_chat(
            system,
            f"Receipt data:\n\n{context}\n\nWrite the 6 tagged insights now:",
            max_new_tokens=450
        )
        return result or ""
    except Exception as e:
        print(f"Qwen insights error: {e}")
        return ""


def format_insight_card(line: str) -> str:
    tag_config = {
        "[SPEND]":    ("wallet",       ""),
        "[STORE]":    ("store",        "info"),
        "[ITEM]":     ("package",      "info"),
        "[CATEGORY]": ("layers",       "warn"),
        "[HABIT]":    ("repeat",       "warn"),
        "[ACTION]":   ("circle-check", ""),
    }
    for tag, (ico_name, kind) in tag_config.items():
        if line.upper().startswith(tag):
            text = line[len(tag):].strip()
            return (
                f'<div class="insight-card {kind}">'
                f'{icon(ico_name, 16)} <strong>{tag}</strong> {text}</div>'
            )
    return f'<div class="insight-card info">{icon("info", 16)} {line}</div>'

# ── image helpers ─────────────────────────────────────────────────────────────

def dark(fig, height=340):
    fig.update_layout(**DARK_LAYOUT, height=height)
    return fig


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(map(str, value)) if value else None
    return str(value).strip()


def parse_price(value):
    if value is None:
        return 0.0
    try:
        matches = re.findall(r"\d+\.\d+|\d+", str(value).replace(",", ""))
        return float(matches[-1]) if matches else 0.0
    except Exception:
        return 0.0


def load_image(image_path, max_size=3000):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w, h  = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def load_pil_image(pil_img: Image.Image, max_size=3000) -> Image.Image:
    image = pil_img.convert("RGB")
    image = ImageOps.exif_transpose(image)
    w, h  = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    return image


def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / width),  1000)),
        max(0, min(int(1000 * y0 / height), 1000)),
        max(0, min(int(1000 * x1 / width),  1000)),
        max(0, min(int(1000 * y1 / height), 1000)),
    ]

# ── OCR ───────────────────────────────────────────────────────────────────────

def paddle_to_words_boxes(image: Image.Image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    image.save(tmp_path, format="JPEG")
    try:
        result = ocr.ocr(tmp_path, cls=False)
    except Exception as e:
        os.unlink(tmp_path)
        raise RuntimeError(f"PaddleOCR failed: {e}")
    os.unlink(tmp_path)
    words, boxes = [], []
    if not result or result[0] is None:
        return words, boxes
    for line in result[0]:
        box  = line[0]
        text = line[1][0].strip()
        if not text:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        split_words = text.split()
        word_width  = max(x1 - x0, 1) / len(split_words)
        for i, word in enumerate(split_words):
            words.append(word)
            boxes.append([int(x0 + i * word_width), y0, int(x0 + (i+1) * word_width), y1])
    return words, boxes

# ── LayoutLMv3 inference ──────────────────────────────────────────────────────

def predict_receipt(image: Image.Image):
    """
    Run full pipeline on a PIL Image directly.
    Returns dict with keys: image, results, raw_text, or error.
    """
    image         = preprocess_image(image)
    width, height = image.size
    words, boxes  = paddle_to_words_boxes(image)
    if not words:
        return {"error": "No OCR text detected. Check image quality."}
    raw_text   = " ".join(words)
    norm_boxes = [normalize_bbox(b, width, height) for b in boxes]
    CHUNK_SIZE, OVERLAP = 150, 20
    seen, results, start = set(), [], 0
    while start < len(words):
        end             = min(start + CHUNK_SIZE, len(words))
        chunk_words     = words[start:end]
        chunk_boxes     = norm_boxes[start:end]
        chunk_raw_boxes = boxes[start:end]
        encoding = processor(
            image, chunk_words, boxes=chunk_boxes,
            truncation=True, padding="max_length",
            max_length=MAX_SEQ_LENGTH, return_tensors="pt"
        )
        word_ids = encoding.word_ids(0)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = layoutlm_model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().cpu().tolist()
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            gid = start + word_id
            if gid in seen:
                continue
            seen.add(gid)
            results.append({
                "word":  chunk_words[word_id],
                "label": id2label[predictions[idx]],
                "bbox":  chunk_raw_boxes[word_id],
            })
        if end == len(words):
            break
        start = end - OVERLAP
    return {"image": image, "results": results, "raw_text": raw_text}

# ── field extraction ──────────────────────────────────────────────────────────

def extract_fields(results: list, raw_text: str = "") -> dict:
    structured = {
        "store_name": None, "store_location": None,
        "receipt_total": None, "receipt_no": None,
        "kra_pin": None, "cashier_name": None,
        "receipt_date": None,
        "items": [],
    }

    def collect_entity(start_idx, entity_name):
        collected, idx = [], start_idx
        while idx < len(results):
            lbl = results[idx]["label"]
            if lbl in [f"B-{entity_name}", f"I-{entity_name}"]:
                collected.append(results[idx]["word"]); idx += 1
            else:
                break
        return " ".join(collected), idx

    current_item, i = None, 0
    while i < len(results):
        label = results[i]["label"]
        if label == "B-STORE_NAME":
            value, i = collect_entity(i, "STORE_NAME")
            structured["store_name"] = normalise_store_name(value)
        elif label == "B-STORE_LOCATION":
            value, i = collect_entity(i, "STORE_LOCATION")
            structured["store_location"] = value
        elif label == "B-RECEIPT_TOTAL":
            value, i = collect_entity(i, "RECEIPT_TOTAL")
            structured["receipt_total"] = value
        elif label == "B-RECEIPT_NO":
            value, i = collect_entity(i, "RECEIPT_NO")
            structured["receipt_no"] = value
        elif label == "B-KRA_PIN":
            value, i = collect_entity(i, "KRA_PIN")
            structured["kra_pin"] = value
        elif label == "B-CASHIER_NAME":
            value, i = collect_entity(i, "CASHIER_NAME")
            structured["cashier_name"] = value
        elif label == "B-ITEM_NAME":
            if current_item:
                structured["items"].append(current_item)
            item_name, i = collect_entity(i, "ITEM_NAME")
            current_item = {
                "item_name": item_name, "qty": None,
                "unit_price": None, "line_total": None, "category": None,
            }
        elif label == "B-QTY":
            value, i = collect_entity(i, "QTY")
            if current_item: current_item["qty"] = value
        elif label == "B-UNIT_PRICE":
            value, i = collect_entity(i, "UNIT_PRICE")
            if current_item: current_item["unit_price"] = value
        elif label == "B-LINE_TOTAL":
            value, i = collect_entity(i, "LINE_TOTAL")
            if current_item: current_item["line_total"] = value
        else:
            i += 1
    if current_item:
        structured["items"].append(current_item)

    # extract date from raw OCR text
    if raw_text:
        structured["receipt_date"] = extract_dates_from_text(raw_text)

    return structured


def draw_labeled_image(image, results):
    vis  = image.copy()
    draw = ImageDraw.Draw(vis)
    for r in results:
        if r["label"] == "O":
            continue
        x0, y0, x1, y1 = r["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, max(0, y0 - 12)), r["label"], fill="blue")
    return vis

# ── CSV export helpers ────────────────────────────────────────────────────────

def make_items_csv(items_df: pd.DataFrame, receipts_df: pd.DataFrame) -> bytes:
    """Merge items with receipt metadata and return CSV bytes."""
    if items_df.empty:
        return b""
    merged = items_df.merge(
        receipts_df[["id","store_name","receipt_total","receipt_date","created_at"]],
        left_on="receipt_id", right_on="id", how="left", suffixes=("","_receipt")
    )
    cols = ["store_name","receipt_date","created_at","item_name",
            "qty","unit_price","line_total","category","receipt_total","receipt_id"]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].to_csv(index=False).encode()


def make_receipts_csv(receipts_df: pd.DataFrame) -> bytes:
    if receipts_df.empty:
        return b""
    return receipts_df.to_csv(index=False).encode()


def make_per_receipt_csv(item_df: pd.DataFrame) -> bytes:
    if item_df.empty:
        return b""
    return item_df.to_csv(index=False).encode()

# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def db_save_receipt(user_id: str, structured: dict, image_filename: str) -> str | None:
    try:
        row = {
            "user_id":        user_id,
            "store_name":     structured.get("store_name"),
            "store_location": structured.get("store_location"),
            "receipt_total":  parse_price(structured.get("receipt_total")),
            "receipt_no":     structured.get("receipt_no"),
            "kra_pin":        structured.get("kra_pin"),
            "cashier_name":   structured.get("cashier_name"),
            "receipt_date":   structured.get("receipt_date"),
            "image_filename": image_filename,
        }
        res = supabase.table("receipts").insert(row).execute()
        return res.data[0]["id"] if res.data else None
    except Exception as e:
        st.error(f"DB error saving receipt: {e}")
        return None


def db_save_items(user_id: str, receipt_id: str, items: list):
    if not items:
        return
    try:
        rows = [
            {
                "user_id":    user_id,
                "receipt_id": receipt_id,
                "item_name":  item.get("item_name"),
                "qty":        parse_price(item.get("qty")),
                "unit_price": parse_price(item.get("unit_price")),
                "line_total": parse_price(item.get("line_total")),
                "category":   item.get("category") or "other",
            }
            for item in items
        ]
        supabase.table("items").insert(rows).execute()
    except Exception as e:
        st.error(f"DB error saving items: {e}")


def db_load_receipts(user_id: str) -> pd.DataFrame:
    try:
        res = (
            supabase.table("receipts")
            .select("id,store_name,store_location,receipt_total,receipt_no,"
                    "kra_pin,cashier_name,receipt_date,image_filename,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"DB error loading receipts: {e}")
        return pd.DataFrame()


def db_load_items(user_id: str) -> pd.DataFrame:
    try:
        res = (
            supabase.table("items")
            .select("id,receipt_id,item_name,qty,unit_price,line_total,category")
            .eq("user_id", user_id)
            .execute()
        )
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"DB error loading items: {e}")
        return pd.DataFrame()


def db_load_receipt_items(receipt_id: str) -> pd.DataFrame:
    try:
        res = (
            supabase.table("items")
            .select("item_name,qty,unit_price,line_total,category")
            .eq("receipt_id", receipt_id)
            .execute()
        )
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"DB error loading receipt items: {e}")
        return pd.DataFrame()


def db_delete_receipt(receipt_id: str):
    try:
        supabase.table("receipts").delete().eq("id", receipt_id).execute()
    except Exception as e:
        st.error(f"DB error deleting receipt: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CORE PROCESSING PIPELINE (shared by single, batch, camera)
# ══════════════════════════════════════════════════════════════════════════════

def process_and_save(
    pil_image: Image.Image,
    filename: str,
    user_id: str,
    use_llm_cat: bool,
) -> dict:
    """
    Run the full pipeline on a PIL image and save to Supabase.
    Returns a result dict with keys: ok, filename, structured, error.
    """
    try:
        prediction = predict_receipt(pil_image)
        if "error" in prediction:
            return {"ok": False, "filename": filename, "error": prediction["error"]}

        structured = extract_fields(
            prediction["results"],
            raw_text=prediction.get("raw_text", "")
        )
        items = structured.get("items", [])

        # categorise
        for item in items:
            item["category"] = categorize_item(item.get("item_name"), use_llm=use_llm_cat)

        # save image to Drive
        save_path = Path(UPLOAD_DIR) / f"{user_id}_{filename}"
        pil_image.save(str(save_path), format="JPEG")

        # save to Supabase
        receipt_id = db_save_receipt(user_id, structured, save_path.name)
        if receipt_id:
            db_save_items(user_id, receipt_id, items)

        return {
            "ok":         True,
            "filename":   filename,
            "structured": structured,
            "prediction": prediction,
            "receipt_id": receipt_id,
        }
    except Exception as e:
        return {"ok": False, "filename": filename, "error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def auth_register(username: str, password: str) -> tuple[bool, str]:
    fake_email = f"{username.lower().strip()}@receiptapp.local"
    try:
        res = supabase.auth.sign_up({"email": fake_email, "password": password})
        if not res.user:
            return False, "Registration failed. Try a different username."
        supabase.table("profiles").insert({
            "id":       res.user.id,
            "username": username.strip(),
        }).execute()
        return True, "Account created! Please log in."
    except Exception as e:
        msg = str(e)
        if "already registered" in msg.lower() or "duplicate" in msg.lower():
            return False, "Username already taken."
        return False, f"Error: {msg}"


def auth_login(username: str, password: str) -> tuple[bool, str]:
    fake_email = f"{username.lower().strip()}@receiptapp.local"
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": fake_email, "password": password}
        )
        if not res.user:
            return False, "Invalid username or password."
        profile_res = (
            supabase.table("profiles")
            .select("*").eq("id", res.user.id).single().execute()
        )
        st.session_state["user"]    = res.user
        st.session_state["profile"] = profile_res.data
        return True, "Logged in."
    except Exception:
        return False, "Invalid username or password."


def auth_logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state["user"]    = None
    st.session_state["profile"] = None
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def render_auth():
    st.markdown(
        f'<div class="auth-card">'
        f'<div class="auth-title">{icon("receipt", 22)} Receipt Intelligence</div>'
        f'<div class="auth-sub">AI-powered spending analytics from your receipts</div>',
        unsafe_allow_html=True
    )
    tab_login, tab_register = st.tabs(["Log In", "Create Account"])
    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In", type="primary", use_container_width=True):
            if username and password:
                ok, msg = auth_login(username, password)
                if ok:
                    st.success(msg); st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Enter your username and password.")
    with tab_register:
        new_user  = st.text_input("Choose a username", key="reg_user")
        new_pass  = st.text_input("Choose a password", type="password", key="reg_pass")
        new_pass2 = st.text_input("Confirm password",  type="password", key="reg_pass2")
        if st.button("Create Account", type="primary", use_container_width=True):
            if not new_user or not new_pass:
                st.warning("Fill in all fields.")
            elif new_pass != new_pass2:
                st.error("Passwords do not match.")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                ok, msg = auth_register(new_user, new_pass)
                st.success(msg) if ok else st.error(msg)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def render_app():
    user_id  = st.session_state["user"].id
    username = st.session_state["profile"]["username"]

    receipts_df = db_load_receipts(user_id)
    items_df    = db_load_items(user_id)

    if not receipts_df.empty and "store_name" in receipts_df.columns:
        receipts_df["store_name"] = receipts_df["store_name"].map(
            lambda v: normalise_store_name(v) if pd.notna(v) else v
        )

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(icon_label("receipt", "Receipt Intelligence", size=20), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            f'{icon("user-circle", 16)} <span style="color:#e0e6f0;font-size:13px">'
            f'Logged in as <strong>{username}</strong></span>',
            unsafe_allow_html=True
        )
        st.markdown("")
        page = st.radio(
            "Navigate",
            ["Dashboard", "Upload Receipt", "Batch Upload",
             "Camera Capture", "Receipt Explorer"],
            label_visibility="collapsed",
            key="page",
        )
        st.markdown("---")
        st.markdown(icon_label("folder-open", f"{len(receipts_df)} receipts", size=14), unsafe_allow_html=True)
        st.markdown(icon_label("cpu",         "LayoutLMv3",                   size=14), unsafe_allow_html=True)
        # Qwen status — show whether it loaded successfully
        qwen_status = "Qwen2.5-1.5B ✓" if qwen_available() else "Qwen2.5-1.5B (CPU — slow)"
        qwen_icon   = "bot" if qwen_available() else "alert-triangle"
        st.markdown(icon_label(qwen_icon, qwen_status, size=14), unsafe_allow_html=True)
        st.markdown(icon_label("scan-text",   "PaddleOCR",  size=14), unsafe_allow_html=True)
        st.markdown(icon_label("database",    "Supabase",   size=14), unsafe_allow_html=True)
        device_label = f"{'GPU' if HAS_GPU else 'CPU'} · {'HF Space' if IS_HF_SPACE else 'Colab'}"
        st.markdown(icon_label("server", device_label, size=14), unsafe_allow_html=True)

        # free tier notice
        if IS_HF_SPACE and not HAS_GPU:
            st.markdown("---")
            st.markdown(
                f'<div style="background:#2a1f0e;border:1px solid #e8a838;border-radius:8px;'
                f'padding:10px 12px;font-size:12px;color:#e8a838;">'
                f'{icon("alert-triangle", 13)} Running on free CPU tier. '
                f'Qwen AI features may be slow.</div>',
                unsafe_allow_html=True
            )
        st.markdown("---")
        if st.button("Log Out", use_container_width=True):
            auth_logout()

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════

    if page == "Dashboard":
        st.markdown(
            f'<h2 style="color:#e8edf5">{icon("layout-dashboard", 28)} '
            f'{username}\'s Dashboard</h2>',
            unsafe_allow_html=True
        )
        st.markdown('<p style="color:#8892a4">AI-powered spending analytics from your receipts</p>',
                    unsafe_allow_html=True)

        if receipts_df.empty:
            st.info("No receipts yet. Upload your first receipt to get started.")
        else:
            total_spend    = receipts_df["receipt_total"].sum()
            total_receipts = len(receipts_df)
            avg_basket     = receipts_df["receipt_total"].mean()
            total_items    = int(items_df["qty"].sum()) if not items_df.empty else 0
            fav_store      = receipts_df["store_name"].value_counts().idxmax() \
                             if receipts_df["store_name"].notna().any() else "—"

            # ── extra metrics ─────────────────────────────────────────────────
            # largest single receipt
            max_receipt     = receipts_df["receipt_total"].max()
            max_receipt_store = receipts_df.loc[
                receipts_df["receipt_total"].idxmax(), "store_name"
            ] if not receipts_df.empty else "—"

            # unique stores visited
            unique_stores = receipts_df["store_name"].nunique()

            # unique items ever bought
            unique_items = items_df["item_name"].nunique() if not items_df.empty else 0

            # avg items per receipt
            avg_items_per_receipt = (
                len(items_df) / total_receipts if total_receipts > 0 else 0
            )

            # category diversity (how many categories have been bought from)
            cat_diversity = items_df["category"].nunique() if not items_df.empty else 0

            tiles_html = '<div class="metric-icon-row">'
            tiles_html += metric_tile("wallet",        "Total Spend",       f"KES {total_spend:,.0f}")
            tiles_html += metric_tile("receipt",       "Receipts",          str(total_receipts))
            tiles_html += metric_tile("shopping-cart", "Avg Basket",        f"KES {avg_basket:,.0f}")
            tiles_html += metric_tile("package",       "Items Bought",      str(total_items))
            tiles_html += metric_tile("store",         "Stores Visited",    str(unique_stores))
            tiles_html += '</div>'
            tiles_html += '<div class="metric-icon-row">'
            tiles_html += metric_tile("star",          "Favourite Store",   fav_store)
            tiles_html += metric_tile("zap",           "Biggest Receipt",   f"KES {max_receipt:,.0f}")
            tiles_html += metric_tile("list",          "Unique Items",      str(unique_items))
            tiles_html += metric_tile("layers",        "Categories Used",   str(cat_diversity))
            tiles_html += metric_tile("bar-chart",     "Items / Receipt",   f"{avg_items_per_receipt:.1f}")
            tiles_html += '</div>'
            st.markdown(tiles_html, unsafe_allow_html=True)

            # biggest receipt callout
            st.markdown(
                f'<div class="insight-card info">'
                f'{icon("zap", 16)} '
                f'Your largest receipt was <strong>KES {max_receipt:,.0f}</strong> '
                f'at <strong>{max_receipt_store}</strong>.</div>',
                unsafe_allow_html=True
            )

            # CSV downloads
            col_dl1, col_dl2, _ = st.columns([1, 1, 2])
            with col_dl1:
                csv_items = make_items_csv(items_df, receipts_df)
                if csv_items:
                    st.download_button(
                        label="Download All Items CSV",
                        data=csv_items,
                        file_name=f"items_{username}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            with col_dl2:
                csv_receipts = make_receipts_csv(receipts_df)
                if csv_receipts:
                    st.download_button(
                        label="Download Receipts CSV",
                        data=csv_receipts,
                        file_name=f"receipts_{username}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            st.markdown("---")

            tab_spend, tab_products, tab_categories, tab_time, tab_insights = st.tabs([
                "Store Analytics", "Product Analytics",
                "Category Breakdown", "Time Analytics", "AI Insights",
            ])

            with tab_spend:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(icon_label("bar-chart-2", "**Spend by Store**"), unsafe_allow_html=True)
                    spend_by_store = (
                        receipts_df.groupby("store_name")["receipt_total"]
                        .sum().reset_index().sort_values("receipt_total", ascending=False)
                    )
                    fig = px.bar(
                        spend_by_store, x="store_name", y="receipt_total",
                        labels={"store_name": "Store", "receipt_total": "Total Spend (KES)"},
                        color="receipt_total", color_continuous_scale="teal",
                    )
                    dark(fig)
                    fig.update_layout(showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    st.markdown(icon_label("pie-chart", "**Visit Distribution**"), unsafe_allow_html=True)
                    visit_counts = receipts_df["store_name"].value_counts().reset_index()
                    visit_counts.columns = ["store_name", "visits"]
                    fig2 = px.pie(
                        visit_counts, names="store_name", values="visits", hole=0.45,
                        color_discrete_sequence=list(CATEGORY_COLORS.values()),
                    )
                    dark(fig2)
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown(icon_label("table", "**Store Summary**"), unsafe_allow_html=True)
                store_summary = (
                    receipts_df.groupby("store_name")
                    .agg(visits=("id","count"),
                         total_spend=("receipt_total","sum"),
                         avg_basket=("receipt_total","mean"))
                    .sort_values("total_spend", ascending=False).reset_index()
                )
                store_summary["total_spend"] = store_summary["total_spend"].map("KES {:,.0f}".format)
                store_summary["avg_basket"]  = store_summary["avg_basket"].map("KES {:,.0f}".format)
                store_summary.columns = ["Store","Visits","Total Spend","Avg Basket"]
                st.dataframe(store_summary, use_container_width=True, hide_index=True)

            with tab_products:
                if items_df.empty:
                    st.info("No item data yet.")
                else:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(icon_label("trending-up", "**Most Purchased Items**"), unsafe_allow_html=True)
                        top_items = items_df["item_name"].value_counts().head(10).reset_index()
                        top_items.columns = ["item_name", "count"]
                        fig = px.bar(
                            top_items, x="count", y="item_name", orientation="h",
                            labels={"item_name": "", "count": "Times Purchased"},
                            color="count", color_continuous_scale="teal",
                        )
                        dark(fig, height=360)
                        fig.update_layout(yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)

                    with col_b:
                        st.markdown(icon_label("banknote", "**Highest Spend Items**"), unsafe_allow_html=True)
                        top_revenue = (
                            items_df.groupby("item_name")["line_total"]
                            .sum().sort_values(ascending=False).head(10).reset_index()
                        )
                        fig2 = px.bar(
                            top_revenue, x="line_total", y="item_name", orientation="h",
                            labels={"item_name": "", "line_total": "Total Spend (KES)"},
                            color="line_total", color_continuous_scale="purples",
                        )
                        dark(fig2, height=360)
                        fig2.update_layout(yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
                        st.plotly_chart(fig2, use_container_width=True)

                    st.markdown(icon_label("activity", "**Receipt Total Distribution**"), unsafe_allow_html=True)
                    fig3 = px.histogram(
                        receipts_df, x="receipt_total", nbins=20,
                        labels={"receipt_total": "Receipt Total (KES)"},
                        color_discrete_sequence=["#4caf96"],
                    )
                    dark(fig3, height=280)
                    st.plotly_chart(fig3, use_container_width=True)

            with tab_categories:
                if items_df.empty or "category" not in items_df.columns:
                    st.info("No category data yet.")
                else:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(icon_label("layers", "**Spend by Category**"), unsafe_allow_html=True)
                        cat_spend = (
                            items_df.groupby("category")["line_total"]
                            .sum().reset_index().sort_values("line_total", ascending=False)
                        )
                        fig = px.bar(
                            cat_spend, x="line_total", y="category", orientation="h",
                            labels={"category": "", "line_total": "Spend (KES)"},
                            color="category", color_discrete_map=CATEGORY_COLORS,
                        )
                        dark(fig, height=360)
                        fig.update_layout(showlegend=False, yaxis={"categoryorder":"total ascending"})
                        st.plotly_chart(fig, use_container_width=True)

                    with col_b:
                        st.markdown(icon_label("pie-chart", "**Category Share of Basket**"), unsafe_allow_html=True)
                        fig2 = px.pie(
                            cat_spend, names="category", values="line_total", hole=0.4,
                            color="category", color_discrete_map=CATEGORY_COLORS,
                        )
                        dark(fig2, height=360)
                        st.plotly_chart(fig2, use_container_width=True)

                    st.markdown(icon_label("table-2", "**Category Detail**"), unsafe_allow_html=True)
                    cat_detail = (
                        items_df.groupby("category")
                        .agg(items=("item_name","count"),
                             unique_items=("item_name","nunique"),
                             total_spend=("line_total","sum"),
                             avg_price=("unit_price","mean"))
                        .sort_values("total_spend", ascending=False).reset_index()
                    )
                    cat_detail["total_spend"] = cat_detail["total_spend"].map("KES {:,.0f}".format)
                    cat_detail["avg_price"]   = cat_detail["avg_price"].map("KES {:,.0f}".format)
                    cat_detail.columns = ["Category","Item Lines","Unique Items","Total Spend","Avg Unit Price"]
                    st.dataframe(cat_detail, use_container_width=True, hide_index=True)

            with tab_time:
                # ── prep: parse receipt_date into datetime ────────────────────
                dated = receipts_df[receipts_df["receipt_date"].notna()].copy()
                if dated.empty:
                    st.info("No date data yet. Upload receipts with readable dates to unlock time analytics.")
                else:
                    dated["date"]       = pd.to_datetime(dated["receipt_date"], errors="coerce")
                    dated               = dated.dropna(subset=["date"])
                    dated["week"]       = dated["date"].dt.to_period("W").apply(lambda p: p.start_time)
                    dated["month"]      = dated["date"].dt.to_period("M").apply(lambda p: p.start_time)
                    dated["dow"]        = dated["date"].dt.day_name()
                    dated["dom"]        = dated["date"].dt.day          # day of month
                    dated["hour_dummy"] = dated["date"].dt.strftime("%Y-%m-%d")

                    # ── time-based metric tiles ───────────────────────────────
                    first_date  = dated["date"].min().strftime("%d %b %Y")
                    last_date   = dated["date"].max().strftime("%d %b %Y")
                    date_range  = (dated["date"].max() - dated["date"].min()).days
                    n_dated     = len(dated)

                    # avg days between trips
                    sorted_dates = dated["date"].sort_values().reset_index(drop=True)
                    if len(sorted_dates) > 1:
                        gaps     = sorted_dates.diff().dropna().dt.days
                        avg_gap  = gaps.mean()
                        avg_gap_str = f"{avg_gap:.1f} days"
                    else:
                        avg_gap_str = "—"

                    # busiest day of week
                    dow_counts  = dated["dow"].value_counts()
                    busiest_dow = dow_counts.idxmax() if not dow_counts.empty else "—"

                    # busiest month
                    month_spend = dated.groupby("month")["receipt_total"].sum()
                    busiest_month = month_spend.idxmax().strftime("%b %Y") if not month_spend.empty else "—"

                    # spend velocity (KES per day over tracked period)
                    spend_velocity = (dated["receipt_total"].sum() / date_range) if date_range > 0 else 0

                    # biggest single-day spend
                    day_spend   = dated.groupby("date")["receipt_total"].sum()
                    biggest_day = day_spend.max()
                    biggest_day_date = day_spend.idxmax().strftime("%d %b %Y") if not day_spend.empty else "—"

                    tiles_html = '<div class="metric-icon-row">'
                    tiles_html += metric_tile("calendar",      "First Receipt",    first_date)
                    tiles_html += metric_tile("calendar-check","Last Receipt",     last_date)
                    tiles_html += metric_tile("timer",         "Avg Trip Gap",     avg_gap_str)
                    tiles_html += metric_tile("trending-up",   "Daily Spend Rate", f"KES {spend_velocity:,.0f}")
                    tiles_html += metric_tile("zap",           "Busiest Day",      busiest_dow)
                    tiles_html += metric_tile("crown",         "Peak Month",       busiest_month)
                    tiles_html += '</div>'
                    st.markdown(tiles_html, unsafe_allow_html=True)

                    # biggest single day callout
                    st.markdown(
                        f'<div class="insight-card info">'
                        f'{icon("flame", 16)} '
                        f'Your biggest shopping day was <strong>{biggest_day_date}</strong> '
                        f'with <strong>KES {biggest_day:,.0f}</strong> spent.</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown("---")

                    # ── spend over time ───────────────────────────────────────
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(icon_label("trending-up", "**Monthly Spend**"), unsafe_allow_html=True)
                        monthly = dated.groupby("month")["receipt_total"].sum().reset_index()
                        monthly.columns = ["month", "spend"]
                        monthly["month_str"] = monthly["month"].dt.strftime("%b %Y")
                        fig_month = px.bar(
                            monthly, x="month_str", y="spend",
                            labels={"month_str": "Month", "spend": "Total Spend (KES)"},
                            color="spend", color_continuous_scale="teal",
                        )
                        dark(fig_month)
                        fig_month.update_layout(
                            coloraxis_showscale=False,
                            xaxis={"categoryorder": "array", "categoryarray": monthly["month_str"].tolist()}
                        )
                        st.plotly_chart(fig_month, use_container_width=True)

                    with col_b:
                        st.markdown(icon_label("trending-up", "**Weekly Spend Trend**"), unsafe_allow_html=True)
                        weekly = dated.groupby("week")["receipt_total"].sum().reset_index()
                        weekly.columns = ["week", "spend"]
                        weekly["week_str"] = weekly["week"].dt.strftime("%d %b")
                        fig_week = px.line(
                            weekly, x="week_str", y="spend",
                            labels={"week_str": "Week starting", "spend": "Spend (KES)"},
                            markers=True,
                            color_discrete_sequence=["#4caf96"],
                        )
                        dark(fig_week)
                        fig_week.update_layout(
                            xaxis={"categoryorder": "array", "categoryarray": weekly["week_str"].tolist()}
                        )
                        st.plotly_chart(fig_week, use_container_width=True)

                    # ── day of week patterns ──────────────────────────────────
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.markdown(icon_label("calendar", "**Shopping by Day of Week**"), unsafe_allow_html=True)
                        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                        dow_data  = (
                            dated.groupby("dow")
                            .agg(trips=("id","count"), total_spend=("receipt_total","sum"))
                            .reindex(dow_order).fillna(0).reset_index()
                        )
                        fig_dow = px.bar(
                            dow_data, x="dow", y="trips",
                            labels={"dow": "Day", "trips": "Number of Trips"},
                            color="trips", color_continuous_scale="purples",
                        )
                        dark(fig_dow, height=300)
                        fig_dow.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_dow, use_container_width=True)

                    with col_d:
                        st.markdown(icon_label("banknote", "**Avg Spend by Day of Week**"), unsafe_allow_html=True)
                        dow_avg = (
                            dated.groupby("dow")["receipt_total"]
                            .mean().reindex(dow_order).fillna(0).reset_index()
                        )
                        dow_avg.columns = ["dow", "avg_spend"]
                        fig_dow_avg = px.bar(
                            dow_avg, x="dow", y="avg_spend",
                            labels={"dow": "Day", "avg_spend": "Avg Spend (KES)"},
                            color="avg_spend", color_continuous_scale="teal",
                        )
                        dark(fig_dow_avg, height=300)
                        fig_dow_avg.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_dow_avg, use_container_width=True)

                    # ── cumulative spend curve ────────────────────────────────
                    st.markdown(icon_label("activity", "**Cumulative Spend Over Time**"), unsafe_allow_html=True)
                    cumulative = (
                        dated.sort_values("date")
                        .assign(cumulative=lambda df: df["receipt_total"].cumsum())
                    )
                    fig_cum = px.area(
                        cumulative, x="date", y="cumulative",
                        labels={"date": "Date", "cumulative": "Cumulative Spend (KES)"},
                        color_discrete_sequence=["#4caf96"],
                    )
                    dark(fig_cum, height=280)
                    fig_cum.update_traces(line_color="#4caf96", fillcolor="rgba(76,175,150,0.15)")
                    st.plotly_chart(fig_cum, use_container_width=True)

                    # ── trip gap timeline ─────────────────────────────────────
                    st.markdown(icon_label("timer", "**Days Between Shopping Trips**"), unsafe_allow_html=True)
                    if len(sorted_dates) > 1:
                        gap_df = pd.DataFrame({
                            "date": sorted_dates[1:].values,
                            "gap":  gaps.values,
                        })
                        fig_gap = px.bar(
                            gap_df, x="date", y="gap",
                            labels={"date": "Date", "gap": "Days Since Last Trip"},
                            color="gap",
                            color_continuous_scale=[[0,"#4caf96"],[0.5,"#e8a838"],[1,"#e85858"]],
                        )
                        dark(fig_gap, height=260)
                        fig_gap.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_gap, use_container_width=True)
                    else:
                        st.caption("Need at least 2 dated receipts to show trip gaps.")

                    # ── store loyalty over time ───────────────────────────────
                    st.markdown(icon_label("store", "**Store Visits Over Time**"), unsafe_allow_html=True)
                    store_time = (
                        dated.groupby(["month", "store_name"])
                        .size().reset_index(name="visits")
                    )
                    store_time["month_str"] = store_time["month"].dt.strftime("%b %Y")
                    fig_sl = px.bar(
                        store_time, x="month_str", y="visits", color="store_name",
                        labels={"month_str": "Month", "visits": "Visits", "store_name": "Store"},
                        color_discrete_sequence=list(CATEGORY_COLORS.values()),
                        barmode="stack",
                    )
                    dark(fig_sl, height=300)
                    fig_sl.update_layout(
                        xaxis={"categoryorder": "array", "categoryarray": store_time["month_str"].unique().tolist()}
                    )
                    st.plotly_chart(fig_sl, use_container_width=True)

                    # ── receipts without dates warning ────────────────────────
                    no_date_count = len(receipts_df) - len(dated)
                    if no_date_count > 0:
                        st.caption(
                            f"ℹ️ {no_date_count} receipt(s) had no extractable date and are excluded from time charts."
                        )

            with tab_insights:
                st.markdown(icon_label("sparkles", "**Qwen2.5 Spending Insights**"), unsafe_allow_html=True)

                if IS_HF_SPACE and not HAS_GPU:
                    st.markdown(
                        f'<div class="insight-card warn">'
                        f'{icon("alert-triangle", 16)} {QWEN_WARNING}</div>',
                        unsafe_allow_html=True
                    )

                st.caption("Powered by Qwen2.5-1.5B. References your actual stores, items, and patterns.")
                if st.button("Generate AI Insights", type="primary"):
                    if not qwen_available():
                        with st.spinner("Loading Qwen2.5 — this may take several minutes on CPU…"):
                            get_qwen()  # trigger lazy load

                    if not qwen_available():
                        st.error(
                            "Qwen2.5 could not be loaded on this hardware. "
                            "This feature requires a GPU Space or sufficient RAM. "
                            "All other app features work normally."
                        )
                    else:
                        with st.spinner("Qwen2.5 is analysing your receipts…"):
                            insights_text = qwen_generate_insights(receipts_df, items_df)
                        if insights_text:
                            for line in insights_text.strip().split("\n"):
                                line = line.strip()
                                if line:
                                    st.markdown(format_insight_card(line), unsafe_allow_html=True)
                            with st.expander("View data fed to Qwen"):
                                st.code(build_insights_context(receipts_df, items_df), language="text")
                        else:
                            st.info("Could not generate insights. Add more receipts and try again.")
                else:
                    st.caption("Click the button above to generate insights.")

                st.markdown("---")
                st.markdown(icon_label("search", "**Search Receipts**"), unsafe_allow_html=True)
                query = st.text_input("Search items or stores", placeholder="e.g. Milk, Naivas…")
                if query and not items_df.empty:
                    q          = query.lower()
                    item_hits  = items_df[items_df["item_name"].str.lower().str.contains(q, na=False)]
                    store_hits = receipts_df[receipts_df["store_name"].str.lower().str.contains(q, na=False)]
                    if not item_hits.empty:
                        st.markdown(f"**Items matching '{query}'**")
                        st.dataframe(
                            item_hits[["receipt_id","item_name","qty",
                                       "unit_price","line_total","category"]].head(20),
                            use_container_width=True, hide_index=True
                        )
                    if not store_hits.empty:
                        st.markdown(f"**Receipts from stores matching '{query}'**")
                        st.dataframe(
                            store_hits[["id","store_name","receipt_total","receipt_date","created_at"]].head(20),
                            use_container_width=True, hide_index=True
                        )

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE UPLOAD
    # ══════════════════════════════════════════════════════════════════════════

    elif page == "Upload Receipt":
        st.markdown(
            f'<h2 style="color:#e8edf5">{icon("upload", 28)} Upload Receipt</h2>',
            unsafe_allow_html=True
        )
        st.markdown('<p style="color:#8892a4">Upload a single receipt image</p>',
                    unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a receipt image", type=["jpg","jpeg","png"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            pil_img            = Image.open(uploaded_file)
            col_prev, col_info = st.columns([1, 1])
            with col_prev:
                st.markdown(icon_label("image", "**Preview**"), unsafe_allow_html=True)
                st.image(pil_img, use_column_width=True)
            with col_info:
                st.markdown(icon_label("info", "**File Info**"), unsafe_allow_html=True)
                st.write(f"**Name:** {uploaded_file.name}")
                st.write(f"**Size:** {pil_img.size[0]} × {pil_img.size[1]} px")
                qwen_help = QWEN_WARNING if (IS_HF_SPACE and not HAS_GPU) else "Slower but smarter. Turn off for faster processing."
                use_llm_cat = st.toggle("Use Qwen2.5 for categorisation", value=HAS_GPU, help=qwen_help)
                if IS_HF_SPACE and not HAS_GPU and use_llm_cat:
                    st.warning(QWEN_WARNING)
                run_btn = st.button("Run AI Extraction", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Processing…"):
                    result = process_and_save(
                        load_pil_image(pil_img),
                        uploaded_file.name,
                        user_id,
                        use_llm_cat,
                    )
                if not result["ok"]:
                    st.error(f"Failed: {result['error']}")
                else:
                    st.success("Receipt processed and saved!")
                    structured = result["structured"]
                    prediction = result["prediction"]

                    col_img, col_data = st.columns([1, 1])
                    with col_img:
                        st.markdown(icon_label("scan", "**Annotated Receipt**"), unsafe_allow_html=True)
                        vis = draw_labeled_image(prediction["image"], prediction["results"])
                        st.image(vis, use_column_width=True)
                    with col_data:
                        st.markdown(icon_label("file-text", "**Extracted Fields**"), unsafe_allow_html=True)
                        field_defs = [
                            ("store",     "Store",     structured.get("store_name")),
                            ("map-pin",   "Location",  structured.get("store_location")),
                            ("wallet",    "Total",     structured.get("receipt_total")),
                            ("hash",      "Receipt #", structured.get("receipt_no")),
                            ("key-round", "KRA PIN",   structured.get("kra_pin")),
                            ("user",      "Cashier",   structured.get("cashier_name")),
                            ("calendar",  "Date",      structured.get("receipt_date") or "—"),
                        ]
                        for ico_name, label, val in field_defs:
                            st.markdown(
                                f'{icon(ico_name, 14)} **{label}:** {val or "—"}',
                                unsafe_allow_html=True
                            )
                        items = structured.get("items", [])
                        if items:
                            st.markdown("---")
                            st.markdown(
                                icon_label("list", f"**Line Items ({len(items)})**"),
                                unsafe_allow_html=True
                            )
                            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
                    with st.expander("View raw JSON"):
                        st.json(structured)

    # ══════════════════════════════════════════════════════════════════════════
    # BATCH UPLOAD
    # ══════════════════════════════════════════════════════════════════════════

    elif page == "Batch Upload":
        st.markdown(
            f'<h2 style="color:#e8edf5">{icon("layers", 28)} Batch Upload</h2>',
            unsafe_allow_html=True
        )
        st.markdown('<p style="color:#8892a4">Upload multiple receipts at once or a ZIP file</p>',
                    unsafe_allow_html=True)

        qwen_help_batch = QWEN_WARNING if (IS_HF_SPACE and not HAS_GPU) else "Off by default for batch — much faster with embedding categorisation."
        use_llm_cat = st.toggle("Use Qwen2.5 for categorisation", value=False, help=qwen_help_batch)
        if IS_HF_SPACE and not HAS_GPU and use_llm_cat:
            st.warning(QWEN_WARNING)

        batch_tab_multi, batch_tab_zip = st.tabs(["Multiple Images", "ZIP File"])

        # ── multiple images ───────────────────────────────────────────────────
        with batch_tab_multi:
            uploaded_files = st.file_uploader(
                "Choose receipt images (select multiple)",
                type=["jpg","jpeg","png"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="batch_multi"
            )

            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} file(s) selected**")
                if st.button("Process All", type="primary", key="batch_multi_btn"):
                    results_log = []
                    progress    = st.progress(0, text="Starting…")
                    status_area = st.container()

                    for idx, uf in enumerate(uploaded_files):
                        progress.progress(
                            (idx + 1) / len(uploaded_files),
                            text=f"Processing {uf.name} ({idx+1}/{len(uploaded_files)})…"
                        )
                        pil_img = Image.open(uf)
                        result  = process_and_save(
                            load_pil_image(pil_img), uf.name, user_id, use_llm_cat
                        )
                        results_log.append(result)

                        with status_area:
                            css = "batch-ok" if result["ok"] else "batch-fail"
                            ico = "check-circle" if result["ok"] else "x-circle"
                            msg = (
                                f'{result["structured"].get("store_name") or "Unknown"} — '
                                f'KES {parse_price(result["structured"].get("receipt_total")):,.0f} — '
                                f'Date: {result["structured"].get("receipt_date") or "—"}'
                                if result["ok"] else result["error"]
                            )
                            st.markdown(
                                f'<div class="batch-row {css}">'
                                f'{icon(ico, 16)} <strong>{uf.name}</strong> — {msg}</div>',
                                unsafe_allow_html=True
                            )

                    progress.empty()
                    ok_count   = sum(1 for r in results_log if r["ok"])
                    fail_count = len(results_log) - ok_count
                    st.success(f"Done — {ok_count} succeeded, {fail_count} failed.")

        # ── zip file ──────────────────────────────────────────────────────────
        with batch_tab_zip:
            zip_file = st.file_uploader(
                "Choose a ZIP file of receipt images",
                type=["zip"],
                label_visibility="collapsed",
                key="batch_zip"
            )

            if zip_file:
                with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                    image_names = [
                        n for n in zf.namelist()
                        if n.lower().endswith((".jpg",".jpeg",".png"))
                        and not n.startswith("__MACOSX")
                    ]

                st.markdown(f"**{len(image_names)} image(s) found in ZIP**")
                if st.button("Process ZIP", type="primary", key="batch_zip_btn"):
                    zip_file.seek(0)
                    results_log = []
                    progress    = st.progress(0, text="Starting…")
                    status_area = st.container()

                    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                        for idx, name in enumerate(image_names):
                            progress.progress(
                                (idx + 1) / len(image_names),
                                text=f"Processing {name} ({idx+1}/{len(image_names)})…"
                            )
                            with zf.open(name) as img_file:
                                pil_img = Image.open(img_file)
                                pil_img.load()
                            fname  = Path(name).name
                            result = process_and_save(
                                load_pil_image(pil_img), fname, user_id, use_llm_cat
                            )
                            results_log.append(result)

                            with status_area:
                                css = "batch-ok" if result["ok"] else "batch-fail"
                                ico = "check-circle" if result["ok"] else "x-circle"
                                msg = (
                                    f'{result["structured"].get("store_name") or "Unknown"} — '
                                    f'KES {parse_price(result["structured"].get("receipt_total")):,.0f} — '
                                    f'Date: {result["structured"].get("receipt_date") or "—"}'
                                    if result["ok"] else result["error"]
                                )
                                st.markdown(
                                    f'<div class="batch-row {css}">'
                                    f'{icon(ico, 16)} <strong>{fname}</strong> — {msg}</div>',
                                    unsafe_allow_html=True
                                )

                    progress.empty()
                    ok_count   = sum(1 for r in results_log if r["ok"])
                    fail_count = len(results_log) - ok_count
                    st.success(f"Done — {ok_count} succeeded, {fail_count} failed.")

    # ══════════════════════════════════════════════════════════════════════════
    # CAMERA CAPTURE
    # ══════════════════════════════════════════════════════════════════════════

    elif page == "Camera Capture":
        st.markdown(
            f'<h2 style="color:#e8edf5">{icon("camera", 28)} Camera Capture</h2>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="color:#8892a4">Take a photo of your receipt directly — '
            'works in browser via Colab tunnel</p>',
            unsafe_allow_html=True
        )

        qwen_help_cam = QWEN_WARNING if (IS_HF_SPACE and not HAS_GPU) else "Slower but smarter."
        use_llm_cat = st.toggle("Use Qwen2.5 for categorisation", value=HAS_GPU, help=qwen_help_cam)
        if IS_HF_SPACE and not HAS_GPU and use_llm_cat:
            st.warning(QWEN_WARNING)

        camera_img = st.camera_input("Point camera at receipt and capture")

        if camera_img is not None:
            pil_img = Image.open(camera_img)

            col_prev, col_info = st.columns([1, 1])
            with col_prev:
                st.markdown(icon_label("image", "**Captured Image**"), unsafe_allow_html=True)
                st.image(pil_img, use_column_width=True)
            with col_info:
                st.markdown(icon_label("info", "**Image Info**"), unsafe_allow_html=True)
                st.write(f"**Size:** {pil_img.size[0]} × {pil_img.size[1]} px")
                process_btn = st.button("Process Receipt", type="primary", use_container_width=True)

            if process_btn:
                filename = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                with st.spinner("Processing camera capture…"):
                    result = process_and_save(
                        load_pil_image(pil_img), filename, user_id, use_llm_cat
                    )

                if not result["ok"]:
                    st.error(f"Failed: {result['error']}")
                else:
                    st.success("Receipt captured and saved!")
                    structured = result["structured"]
                    prediction = result["prediction"]

                    col_img, col_data = st.columns([1, 1])
                    with col_img:
                        st.markdown(icon_label("scan", "**Annotated Receipt**"), unsafe_allow_html=True)
                        vis = draw_labeled_image(prediction["image"], prediction["results"])
                        st.image(vis, use_column_width=True)
                    with col_data:
                        st.markdown(icon_label("file-text", "**Extracted Fields**"), unsafe_allow_html=True)
                        field_defs = [
                            ("store",     "Store",     structured.get("store_name")),
                            ("map-pin",   "Location",  structured.get("store_location")),
                            ("wallet",    "Total",     structured.get("receipt_total")),
                            ("hash",      "Receipt #", structured.get("receipt_no")),
                            ("key-round", "KRA PIN",   structured.get("kra_pin")),
                            ("user",      "Cashier",   structured.get("cashier_name")),
                            ("calendar",  "Date",      structured.get("receipt_date") or "—"),
                        ]
                        for ico_name, label, val in field_defs:
                            st.markdown(
                                f'{icon(ico_name, 14)} **{label}:** {val or "—"}',
                                unsafe_allow_html=True
                            )
                        items = structured.get("items", [])
                        if items:
                            st.markdown("---")
                            st.markdown(
                                icon_label("list", f"**Line Items ({len(items)})**"),
                                unsafe_allow_html=True
                            )
                            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
                    with st.expander("View raw JSON"):
                        st.json(structured)

    # ══════════════════════════════════════════════════════════════════════════
    # RECEIPT EXPLORER
    # ══════════════════════════════════════════════════════════════════════════

    elif page == "Receipt Explorer":
        st.markdown(
            f'<h2 style="color:#e8edf5">{icon("search", 28)} Receipt Explorer</h2>',
            unsafe_allow_html=True
        )
        st.markdown('<p style="color:#8892a4">Browse and inspect your saved receipts</p>',
                    unsafe_allow_html=True)

        if receipts_df.empty:
            st.info("No receipts found. Upload some receipts first.")
        else:
            receipts_df["label"] = receipts_df.apply(
                lambda r: (
                    f"{r['store_name'] or 'Unknown'} — "
                    f"{r.get('receipt_date') or str(r['created_at'])[:10]} — "
                    f"KES {r['receipt_total']:,.0f}"
                ),
                axis=1
            )
            selected_label = st.selectbox("Select Receipt", receipts_df["label"].tolist())
            selected_row   = receipts_df[receipts_df["label"] == selected_label].iloc[0]
            receipt_id     = selected_row["id"]

            col_meta, col_items = st.columns([1, 1])

            with col_meta:
                st.markdown(icon_label("file-text", "**Receipt Details**"), unsafe_allow_html=True)
                detail_defs = [
                    ("store",     "Store",        selected_row.get("store_name",     "—")),
                    ("map-pin",   "Location",     selected_row.get("store_location", "—")),
                    ("wallet",    "Total",        f"KES {selected_row['receipt_total']:,.0f}"),
                    ("hash",      "Receipt #",    selected_row.get("receipt_no",     "—")),
                    ("key-round", "KRA PIN",      selected_row.get("kra_pin",        "—")),
                    ("user",      "Cashier",      selected_row.get("cashier_name",   "—")),
                    ("calendar",  "Receipt Date", selected_row.get("receipt_date",   "—") or "—"),
                    ("clock",     "Uploaded",     str(selected_row.get("created_at","—"))[:10]),
                ]
                for ico_name, label, val in detail_defs:
                    st.markdown(
                        f'{icon(ico_name, 14)} **{label}:** {val}',
                        unsafe_allow_html=True
                    )

                img_filename = selected_row.get("image_filename")
                if img_filename:
                    img_path = Path(UPLOAD_DIR) / img_filename
                    if img_path.exists():
                        st.markdown("---")
                        st.markdown(icon_label("image", "**Receipt Image**"), unsafe_allow_html=True)
                        st.image(str(img_path), use_column_width=True)

                st.markdown("---")
                if st.button("Delete this receipt", type="secondary"):
                    db_delete_receipt(receipt_id)
                    st.success("Receipt deleted.")
                    st.rerun()

            with col_items:
                item_df = db_load_receipt_items(receipt_id)
                if not item_df.empty:
                    st.markdown(
                        icon_label("list", f"**Line Items ({len(item_df)})**"),
                        unsafe_allow_html=True
                    )
                    mask = item_df["category"].isna() | (item_df["category"] == "None")
                    item_df.loc[mask, "category"] = (
                        item_df.loc[mask, "item_name"].apply(fast_categorise_item)
                    )

                    if "line_total" in item_df.columns:
                        cat_mini = item_df.groupby("category")["line_total"].sum().reset_index()
                        fig = px.pie(
                            cat_mini, names="category", values="line_total", hole=0.4,
                            color="category", color_discrete_map=CATEGORY_COLORS,
                            title="Category Breakdown",
                        )
                        dark(fig, height=260)
                        fig.update_layout(margin=dict(t=40, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(item_df, use_container_width=True, hide_index=True)

                    # per-receipt CSV download
                    per_csv = make_per_receipt_csv(item_df)
                    store   = selected_row.get("store_name","receipt")
                    date    = selected_row.get("receipt_date") or str(selected_row["created_at"])[:10]
                    st.download_button(
                        label="Download this receipt as CSV",
                        data=per_csv,
                        file_name=f"{store}_{date}.csv".replace(" ","_"),
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info("No line items for this receipt.")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["user"] is None:
    render_auth()
else:
    render_app()
