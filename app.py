"""
Defect Duplicate Checker
– SQLite persistent database
– CSV import/sync (new records added, existing preserved)
– Manual defect entry → instant duplicate check
– Clean dark industrial UI
"""

import re
import sqlite3
import pickle
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# Paths & Constants
# ──────────────────────────────────────────────
DB_PATH       = Path("defects.db")
EMB_PATH      = Path("defects_embeddings.pkl")

EMB_SEMANTIC_THRESHOLD = 0.88
FULL_TEXT_WEIGHT       = 0.40
PROBLEM_TEXT_WEIGHT    = 0.60
MIN_TOKENS             = 3
SENTENCE_MODEL_NAME    = "paraphrase-multilingual-MiniLM-L12-v2"

# ──────────────────────────────────────────────
# NLP helpers
# ──────────────────────────────────────────────
FLOW_GROUPS: Dict[str, Set[str]] = {
    "auth":         {"login","logout","signin","signup","register","otp","verification","verify",
                     "password","pin","biometrics","fingerprint","faceid","2fa","authentication"},
    "messaging":    {"message","chat","send","receive","delivery","delivered","read","unread",
                     "typing","sticker","emoji","gif","media","photo","video","file","document",
                     "forward","reply","delete","unsend","attachment"},
    "calling":      {"call","voice","videocall","video","ringing","ring","answer","decline",
                     "reject","missed","mute","speaker","bluetooth","headset","mic","microphone",
                     "echo","noise"},
    "notification": {"notification","push","badge","sound","vibration","alert","banner"},
    "channel":      {"channel","discovery","explore","search","find","broadcast"},
    "story":        {"story","status","highlight","viewer","reaction"},
    "settings":     {"settings","profile","privacy","account","theme","language","backup",
                     "sync","storage"},
    "permission":   {"permission","camera","contacts","location","microphone","allow","deny",
                     "granted","revoked"},
    "crash":        {"crash","freeze","hang","stuck","lag","slow","anr","unresponsive",
                     "force_close","not_responding","black_screen","white_screen"},
    "payment":      {"payment","purchase","subscription","billing","invoice","refund","card",
                     "wallet","topup","transfer","transaction"},
    "ui":           {"menu","overflow","kebab","tab","button","tap","click","press",
                     "longpress","swipe","scroll","open","close","back","gesture",
                     "layout","overlap","misalign","truncate","cut","hidden"},
}

STOPWORDS = set(
    "a an the and or but if then else when while for to of in on at by with without "
    "from into is are was were be been being this that these those it its as "
    "ve veya ama eger ise degil icin ile bir bu da de".split()
)

IGNORE_REGEXES = [
    r"\b(app\s*)?version\s*[:=]\s*[^\n\r]+",
    r"\bbuild\s*[:=]\s*[^\n\r]+",
    r"\bdevice\s*[:=]\s*[^\n\r]+",
    r"https?://\S+",
    r"\b\d+\.\d+\.\d+(\.\d+)?\b",
    r"\b\d+\b",
]

CONTEXT_TOKENS: Set[str] = {
    "call","log","calllog","history","callhistory","notification","notifications",
    "screen","page","tab","menu","panel","view","list","chat","inbox","feed","home",
    "settings","profile","header","footer","toolbar","bottom","top","briefing",
    "dialer","keypad","dialpad","channel","story","status","highlight","search",
    "result","results","filter","group","contact","contacts",
    "android","ios","iphone","samsung","device","app","application",
}

DEFECT_SIGNALS: List[str] = [
    "wrong","incorrect","invalid","inaccurate","mismatch","mismatched","malformed",
    "missing","disappear","disappeared","not shown","not showing","not display",
    "not displayed","not visible","hidden","gone","not appear","not appearing",
    "not work","not working","broken","fail","failed","fails",
    "unable","cannot","can't","does not","doesn't","won't",
    "no longer","stopped","stop working","crash","crashes","crashed",
    "freeze","frozen","hang","hangs","anr","not responding","force close",
    "black screen","white screen","duplicate","duplicated",
    "not saved","not updated","not synced","not cleared",
    "still showing","showing old","stale","outdated","unexpected","unintended",
    "overlap","overlapping","truncated","cut off","clipped","not centered",
    "receiving","received","not received","still receiving","incorrectly",
    "incorrect duration","wrong duration","incorrect time","wrong time",
    "gorunmuyor","calismiyor","hatali","acilmiyor","kapanmiyor","donuyor",
]

SEPARATOR_RE    = re.compile(r"\s*[-\u2013\u2014|:]\s*")
CONTEXT_PREP_RE = re.compile(
    r"\s+(in|on|at|for|of|within|inside|under|from|during|after|when|while)\s+",
    re.IGNORECASE,
)


def _seg_defect_score(seg: str) -> int:
    sl = seg.lower()
    return sum(1 for sig in DEFECT_SIGNALS if sig in sl)


def extract_problem_text(summary, description) -> str:
    raw_summary = "" if pd.isna(summary) else str(summary).strip()
    raw_desc    = "" if pd.isna(description) else str(description).strip()
    segments    = SEPARATOR_RE.split(raw_summary)
    best_seg    = raw_summary
    if len(segments) > 1:
        scored   = sorted(segments, key=_seg_defect_score, reverse=True)
        best_seg = scored[0].strip()
        if _seg_defect_score(best_seg) == 0:
            best_seg = raw_summary
    prep_split = CONTEXT_PREP_RE.split(best_seg)
    if len(prep_split) >= 3:
        candidate = prep_split[0].strip()
        if _seg_defect_score(candidate) > 0:
            best_seg = candidate
    words         = best_seg.split()
    cleaned_words = [
        w for w in words
        if w.lower().rstrip("s") not in CONTEXT_TOKENS
        or any(sig in w.lower() for sig in DEFECT_SIGNALS)
    ]
    problem_core = " ".join(cleaned_words).strip()
    if len(problem_core.split()) < 3:
        problem_core = best_seg
    desc_sents   = re.split(r"[.\n]", raw_desc)
    defect_sents = []
    for sent in desc_sents[:10]:
        if any(sig in sent.lower() for sig in DEFECT_SIGNALS):
            clean = re.sub(r"\s+", " ", sent).strip()
            if len(clean) > 10:
                defect_sents.append(clean)
        if len(defect_sents) >= 2:
            break
    combined = problem_core
    if defect_sents:
        combined = problem_core + " " + " ".join(defect_sents)
    return combined.strip()


def normalize_text(summary, desc) -> str:
    s = f"{'' if pd.isna(summary) else str(summary)} {'' if pd.isna(desc) else str(desc)}"
    s = s.lower()
    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_problem(raw: str) -> str:
    s = raw.lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\b\d+\.\d+\.\d+\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(norm: str) -> List[str]:
    return [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]


def get_flows(token_set: Set[str]) -> Set[str]:
    return {flow for flow, kws in FLOW_GROUPS.items() if token_set & kws}


def pick_col(cols: List[str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        if any(k in c.lower() for k in keywords):
            return c
    return None


# ──────────────────────────────────────────────
# SQLite layer
# ──────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS defects (
            issue_key   TEXT PRIMARY KEY,
            summary     TEXT,
            description TEXT,
            tester      TEXT,
            norm        TEXT,
            problem     TEXT,
            raw         TEXT,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    return con


def db_count(con) -> int:
    return con.execute("SELECT COUNT(*) FROM defects").fetchone()[0]


def db_last_updated(con) -> str:
    row = con.execute("SELECT MAX(imported_at) FROM defects").fetchone()[0]
    return row or "—"


def db_load_all(con) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM defects", con)


def db_sync_csv(con, raw_csv: str) -> Tuple[int, int]:
    """Insert new rows only. Returns (added, skipped)."""
    df   = pd.read_csv(StringIO(raw_csv), sep=None, engine="python", on_bad_lines="skip")
    cols = list(df.columns)

    key_col     = pick_col(cols, ["issue key","issue_key","key"]) or cols[0]
    summary_col = pick_col(cols, ["summary","title"])             or cols[0]
    desc_col    = pick_col(cols, ["description"])                 or cols[0]
    tester_col  = pick_col(cols, ["assignee","tester","reporter","assigned to","owner"])

    added = skipped = 0
    for _, row in df.iterrows():
        key  = str(row[key_col]).strip()
        summ = "" if pd.isna(row[summary_col]) else str(row[summary_col]).strip()
        desc = "" if pd.isna(row[desc_col])    else str(row[desc_col]).strip()
        test = "" if (tester_col is None or pd.isna(row[tester_col])) else str(row[tester_col]).strip()

        exists = con.execute("SELECT 1 FROM defects WHERE issue_key=?", (key,)).fetchone()
        if exists:
            skipped += 1
            continue

        norm    = normalize_text(summ, desc)
        problem = normalize_problem(extract_problem_text(summ, desc))
        raw     = f"{summ} {desc}".strip()

        con.execute(
            "INSERT INTO defects(issue_key,summary,description,tester,norm,problem,raw) VALUES(?,?,?,?,?,?,?)",
            (key, summ, desc, test, norm, problem, raw)
        )
        added += 1

    con.commit()
    return added, skipped


# ──────────────────────────────────────────────
# Embedding cache (pickle file alongside DB)
# ──────────────────────────────────────────────
def load_emb_cache() -> Dict:
    if EMB_PATH.exists():
        with open(EMB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_emb_cache(cache: Dict):
    with open(EMB_PATH, "wb") as f:
        pickle.dump(cache, f)


def rebuild_missing_embeddings(model, con, cache: Dict) -> Dict:
    df      = db_load_all(con)
    missing = df[~df["issue_key"].isin(cache.keys())]
    if missing.empty:
        return cache
    with st.spinner(f"Encoding {len(missing)} new record(s)…"):
        emb_full    = model.encode(missing["raw"].tolist(),     batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        emb_problem = model.encode(missing["problem"].tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        for i, key in enumerate(missing["issue_key"].tolist()):
            cache[key] = {"full": emb_full[i], "problem": emb_problem[i]}
    save_emb_cache(cache)
    return cache


# ──────────────────────────────────────────────
# Duplicate check — returns best match or None
# ──────────────────────────────────────────────
def check_duplicate(
    query_summary: str,
    query_desc: str,
    con,
    cache: Dict,
    model,
    threshold: float,
) -> Optional[Dict]:
    if not cache:
        return None

    raw     = f"{query_summary} {query_desc}".strip()
    problem = normalize_problem(extract_problem_text(query_summary, query_desc))
    norm    = normalize_text(query_summary, query_desc)
    tok_set = set(tokenize(norm))

    q_full    = model.encode([raw],     normalize_embeddings=True)[0]
    q_problem = model.encode([problem], normalize_embeddings=True)[0]

    q_blend  = FULL_TEXT_WEIGHT * q_full + PROBLEM_TEXT_WEIGHT * q_problem
    q_blend /= (np.linalg.norm(q_blend) or 1.0)

    keys       = list(cache.keys())
    db_full    = np.stack([cache[k]["full"]    for k in keys])
    db_problem = np.stack([cache[k]["problem"] for k in keys])

    db_blend = FULL_TEXT_WEIGHT * db_full + PROBLEM_TEXT_WEIGHT * db_problem
    norms    = np.linalg.norm(db_blend, axis=1, keepdims=True)
    norms    = np.where(norms == 0, 1, norms)
    db_blend = db_blend / norms

    scores_blend = db_blend @ q_blend

    best_score = -1.0
    best_idx   = -1

    for i, key in enumerate(keys):
        sb = float(scores_blend[i])
        if sb < threshold:
            continue
        row_norm = con.execute("SELECT norm FROM defects WHERE issue_key=?", (key,)).fetchone()
        if row_norm:
            db_tok_set = set(tokenize(row_norm[0]))
            db_flows   = get_flows(db_tok_set)
            q_flows    = get_flows(tok_set)
            if db_flows and q_flows and not (db_flows & q_flows):
                continue
        if sb > best_score:
            best_score = sb
            best_idx   = i

    if best_idx == -1:
        return None

    best_key = keys[best_idx]
    row = con.execute(
        "SELECT issue_key, summary, tester, norm FROM defects WHERE issue_key=?",
        (best_key,)
    ).fetchone()

    tok_union = len(set(tokenize(row[3])) | tok_set)
    jac       = len(set(tokenize(row[3])) & tok_set) / tok_union if tok_union > 0 else 0.0
    dup_type  = "Exact" if (norm == row[3] or jac >= 0.90) else "Semantic"

    return {
        "issue_key": row[0],
        "summary":   row[1],
        "tester":    row[2],
        "score":     round(best_score, 3),
        "type":      dup_type,
    }


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading semantic model…")
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(SENTENCE_MODEL_NAME)
    except ImportError:
        return None


# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
def render_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

    .stApp { background: #0d0d0d; color: #e8e8e0; }

    .stTabs [data-baseweb="tab-list"] {
        background: #141414; border-bottom: 2px solid #222; gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
        font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
        color: #555; padding: 0.75rem 1.5rem;
        border-bottom: 2px solid transparent; background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #f0c040 !important; border-bottom: 2px solid #f0c040 !important;
        background: transparent !important;
    }

    .app-title {
        font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.65rem;
        color: #f0c040; letter-spacing: -0.02em; padding: 1.2rem 0 0.15rem;
    }
    .app-sub {
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
        color: #444; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1.6rem;
    }

    .result-dup {
        background: #140f00; border: 1px solid #f0c040; border-left: 4px solid #f0c040;
        border-radius: 4px; padding: 1.3rem 1.6rem;
        font-family: 'JetBrains Mono', monospace; margin-top: 0.5rem;
    }
    .result-ok {
        background: #061008; border: 1px solid #1e5c28; border-left: 4px solid #3ddc5a;
        border-radius: 4px; padding: 1.3rem 1.6rem;
        font-family: 'JetBrains Mono', monospace; margin-top: 0.5rem;
    }
    .result-label { font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; color: #666; margin-bottom: 0.45rem; }
    .result-key   { font-size: 1.35rem; font-weight: 700; color: #f0c040; }
    .result-ok-key { font-size: 1.35rem; font-weight: 700; color: #3ddc5a; }
    .result-summary { font-size: 0.78rem; color: #aaa; margin-top: 0.35rem; line-height: 1.4; }
    .result-meta    { font-size: 0.7rem; color: #555; margin-top: 0.25rem; }

    .pill {
        display: inline-block; font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; font-weight: 700; padding: 2px 9px;
        border-radius: 2px; margin-left: 0.5rem; vertical-align: middle;
    }
    .pill-amber  { background: #2a1e00; border: 1px solid #f0c040; color: #f0c040; }
    .pill-red    { background: #200a0a; border: 1px solid #e05252; color: #e05252; }

    .stTextInput > div > div > input,
    .stTextArea  > div > div > textarea {
        background: #111 !important; border: 1px solid #2a2a2a !important;
        border-radius: 3px !important; color: #e8e8e0 !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.84rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea  > div > div > textarea:focus {
        border-color: #f0c040 !important; box-shadow: none !important;
    }
    label { font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important;
            letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #666 !important; }

    .stButton > button {
        background: #f0c040 !important; color: #0d0d0d !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
        font-weight: 700 !important; letter-spacing: 0.1em !important;
        text-transform: uppercase !important; border: none !important;
        border-radius: 2px !important; padding: 0.55rem 1.4rem !important;
    }
    .stButton > button:hover { background: #ffd060 !important; }

    [data-testid="metric-container"] {
        background: #111; border: 1px solid #1e1e1e; border-radius: 3px; padding: 0.75rem 1rem;
    }
    [data-testid="metric-container"] label {
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.62rem !important;
        letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #444 !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #f0c040 !important;
    }

    [data-testid="stFileUploadDropzone"] {
        background: #111 !important; border: 1px dashed #2a2a2a !important; border-radius: 4px !important;
    }
    .stDataFrame { border: 1px solid #1e1e1e !important; border-radius: 3px; }
    hr { border-color: #1e1e1e !important; }

    .empty-state {
        color: #333; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        border: 1px dashed #1e1e1e; border-radius: 4px; padding: 2.5rem;
        text-align: center; margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Defect Duplicate Checker",
        layout="wide",
        page_icon="⬡",
        initial_sidebar_state="collapsed",
    )
    render_css()

    st.markdown('<div class="app-title">⬡ DEFECT DUPLICATE CHECKER</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Semantic similarity · Persistent SQLite · Multilingual</div>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Run:  pip install sentence-transformers")
        st.stop()

    con   = init_db()
    cache = load_emb_cache()

    if db_count(con) > 0:
        cache = rebuild_missing_embeddings(model, con, cache)

    tab_check, tab_import, tab_db = st.tabs([
        "CHECK DEFECT",
        "IMPORT / SYNC DB",
        "VIEW DATABASE",
    ])

    # ════════════════════════════════
    # TAB 1 — CHECK
    # ════════════════════════════════
    with tab_check:
        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("#### New defect entry")
            threshold = st.slider(
                "Similarity threshold",
                min_value=0.70, max_value=0.99,
                value=float(EMB_SEMANTIC_THRESHOLD), step=0.01,
                help="0.88 default. Lower = more results, higher = stricter.",
            )
            query_summary = st.text_input(
                "Scenario name",
                placeholder="e.g. Wrong name shown in call history",
                key="q_summary",
            )
            query_desc = st.text_area(
                "Scenario steps / description",
                placeholder=(
                    "Steps:\n"
                    "1. Open app\n"
                    "2. Navigate to call history\n"
                    "3. Observe wrong contact name\n\n"
                    "Expected: correct name is displayed"
                ),
                height=170,
                key="q_desc",
            )
            check_btn = st.button("▶  CHECK FOR DUPLICATE", use_container_width=True)

        with col_result:
            st.markdown("#### Result")

            if db_count(con) == 0:
                st.markdown(
                    '<div class="empty-state">DATABASE EMPTY<br><br>'
                    'Go to <b>IMPORT / SYNC DB</b><br>to load your defects first.</div>',
                    unsafe_allow_html=True,
                )
            elif check_btn:
                if not query_summary.strip() and not query_desc.strip():
                    st.warning("Please fill in at least the scenario name.")
                else:
                    with st.spinner("Checking…"):
                        result = check_duplicate(
                            query_summary, query_desc,
                            con, cache, model, threshold
                        )

                    if result is None:
                        st.markdown("""
                        <div class="result-ok">
                            <div class="result-label">Status</div>
                            <div class="result-ok-key">✓ NO DUPLICATE FOUND</div>
                            <div class="result-summary">No matching defect above the threshold.</div>
                            <div class="result-meta">Safe to log as a new issue.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        pill = (
                            f'<span class="pill pill-red">EXACT</span>'
                            if result["type"] == "Exact"
                            else f'<span class="pill pill-amber">SEMANTIC · {result["score"]}</span>'
                        )
                        tester_line = f"<div class='result-meta'>Tester: {result['tester']}</div>" if result["tester"] else ""
                        summary_snip = (result["summary"] or "")[:140]
                        if len(result["summary"] or "") > 140:
                            summary_snip += "…"

                        st.markdown(f"""
                        <div class="result-dup">
                            <div class="result-label">⚠ Already exists in database</div>
                            <div class="result-key">{result['issue_key']}{pill}</div>
                            <div class="result-summary">{summary_snip}</div>
                            {tester_line}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="empty-state">'
                    f'<span style="color:#f0c040;font-size:1.3rem;font-weight:800;">'
                    f'{db_count(con):,}</span><br>records in database<br><br>'
                    f'Fill in the form and press CHECK</div>',
                    unsafe_allow_html=True,
                )

    # ════════════════════════════════
    # TAB 2 — IMPORT
    # ════════════════════════════════
    with tab_import:
        st.markdown("#### Import or sync a CSV into the database")
        st.markdown(
            "<span style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#444;'>"
            "New issue keys are inserted. Existing keys are skipped — no overwrites."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown(" ")

        uploaded = st.file_uploader(
            "Drop CSV here",
            type=["csv"],
            key="import_csv",
            help="Required columns: Issue Key, Summary/Title, Description. Tester/Assignee optional.",
        )

        if uploaded:
            raw_csv = uploaded.getvalue().decode("utf-8", errors="replace")

            with st.expander("Preview (first 5 rows)", expanded=False):
                preview_df = pd.read_csv(
                    StringIO(raw_csv), sep=None, engine="python", on_bad_lines="skip", nrows=5
                )
                st.dataframe(preview_df, use_container_width=True, hide_index=True)

            if st.button("⬆  SYNC INTO DATABASE", use_container_width=False):
                with st.spinner("Importing…"):
                    added, skipped = db_sync_csv(con, raw_csv)
                    cache = rebuild_missing_embeddings(model, con, cache)

                c1, c2, c3 = st.columns(3)
                c1.metric("Added",    added)
                c2.metric("Skipped",  skipped)
                c3.metric("Total DB", db_count(con))

                if added > 0:
                    st.success(f"{added} new defect(s) added.")
                else:
                    st.info("No new records — all issue keys already exist in DB.")

    # ════════════════════════════════
    # TAB 3 — VIEW DB
    # ════════════════════════════════
    with tab_db:
        st.markdown("#### Database overview")

        total = db_count(con)
        last  = db_last_updated(con)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Defects",   f"{total:,}")
        m2.metric("Semantic Model",  "MiniLM-L12-v2")
        m3.metric("Last Import",     str(last)[:16] if last != "—" else "—")

        st.markdown(" ")

        if total == 0:
            st.markdown(
                '<div class="empty-state">No records yet.<br>Import a CSV to get started.</div>',
                unsafe_allow_html=True,
            )
        else:
            df_view = db_load_all(con)[["issue_key","summary","tester","imported_at"]].copy()
            df_view.columns = ["Issue Key", "Summary", "Tester", "Imported At"]
            st.dataframe(df_view, use_container_width=True, hide_index=True, height=400)

            st.download_button(
                label="⬇  Export as CSV",
                data=df_view.to_csv(index=False).encode("utf-8"),
                file_name="defects_export.csv",
                mime="text/csv",
            )

            with st.expander("⚠ Danger Zone", expanded=False):
                st.warning("Permanently deletes ALL records and embeddings from disk.")
                if st.button("🗑  CLEAR ENTIRE DATABASE", type="secondary"):
                    con.execute("DELETE FROM defects")
                    con.commit()
                    if EMB_PATH.exists():
                        EMB_PATH.unlink()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.rerun()


if __name__ == "__main__":
    main()
