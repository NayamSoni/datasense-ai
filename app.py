import hashlib
import html
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd

from llm_agent import ask_llm
from intent_agent import detect_intent
from pandas_agent import statistical_summary, calculate
from query_planner import create_execution_plan, _dimension_mentioned_in_question, DIMENSION_KEYWORDS
from visualization import render_chart, render_manual_chart, render_report_chart
from theme import load_theme
from conversation_memory import contextualize_question, is_follow_up, merge_follow_up_plan
from data_quality import (
    apply_cleaning,
    calculate_health_score,
    cleaning_suggestions,
    profile_data_quality,
)
from insights_engine import (
    generate_business_insights,
    generate_decision_report,
    generate_recommendations,
    generate_report_chart_specs,
    report_to_html,
    report_to_markdown,
)
from feedback_memory import (
    apply_feedback_rules,
    delete_feedback_rule,
    inject_feedback_context,
    is_feedback_message,
    list_feedback_rules,
    propose_feedback_rule,
    relevant_feedback_rules,
    save_feedback_rule,
)
from predictive_modeling import (
    render_data_science_lab,
    render_data_science_lab_intro,
)

try:
    from streamlit_sortables import sort_items
    SORTABLES_AVAILABLE = True
except ImportError:
    SORTABLES_AVAILABLE = False

APP_BUILD = "2026.07.30-DATA-SCIENCE-LAB-UI-R3"
print(f"### APP.PY LOADED - DATASENSE AI {APP_BUILD} ###")
print(f"### SORTABLES_AVAILABLE = {SORTABLES_AVAILABLE} ###")

APP_DIR = Path(__file__).resolve().parent
HERO_IMAGE_PATH = APP_DIR / "assets" / "robot.png"


def chat_avatar(role: str) -> str:
    """Return Streamlit-safe avatars.

    Streamlit 1.45 can fail while decoding local SVG files inside
    ``st.chat_message``. Emoji fallbacks keep chat available on every machine;
    the animated DataSense character will be rendered separately from the chat
    avatar in the next UI milestone.
    """
    return "🧑" if role == "user" else "🤖"


def conversational_reply(question: str) -> str | None:
    """Answer basic conversational messages without involving the data planner."""
    normalized = " ".join(str(question).lower().strip().strip("!?.,").split())

    if normalized in {"hi", "hello", "hey", "hi there", "hello there", "hey there"}:
        return "Hi, how can I help you today?"

    if normalized in {"thanks", "thank you", "thank you so much"}:
        return "You’re welcome! What would you like to explore next?"

    return None


def queue_chat_question() -> None:
    """Queue typed chat input so it is processed before the composer renders."""
    submitted = st.session_state.get("ai_workspace_question")
    if submitted and submitted.strip():
        st.session_state.pending_question = submitted.strip()


DATASET_SESSION_KEYS = (
    "uploaded_file_bytes",
    "uploaded_file_name",
    "uploaded_file_size",
    "active_dataset_id",
    "original_df",
    "active_df",
    "last_analysis_plan",
    "last_analysis_question",
    "pending_feedback_rule",
    "cleaning_log",
)


def handle_dataset_uploader_change() -> None:
    """Persist a new upload, or clear dataset state when its X is clicked."""
    current_file = st.session_state.get("dataset_uploader")

    if current_file is not None:
        st.session_state.uploaded_file_bytes = current_file.getvalue()
        st.session_state.uploaded_file_name = current_file.name
        st.session_state.uploaded_file_size = current_file.size
        return

    for key in DATASET_SESSION_KEYS:
        st.session_state.pop(key, None)

    st.session_state.messages = []

    for key in list(st.session_state):
        if str(key).startswith(
            (
                "insights_",
                "report_",
                "viz_shelves_",
                "chart_style_",
                "predictive_result_",
            )
        ):
            st.session_state.pop(key, None)


def compact_html(markup: str) -> str:
    """Keep Streamlit Markdown from breaking nested raw HTML at blank lines."""
    return "".join(line.strip() for line in markup.splitlines())


# ==========================================================
# STATIC CONTENT - Blending placeholder
# ==========================================================

BLENDING_CONTENT = """
## Dataset Blending

**Coming soon.**

This will let you join and blend multiple datasets together
(e.g. 2-3 related tables) for cross-table analysis - matching
records across files instead of analyzing one dataset at a time.
"""


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="DataSense AI",
    layout="wide"
)

load_theme()

# One restrained typography scale keeps every surface readable and makes the
# interface feel like a single business product.
st.markdown(
    """
    <style>
    :root {
        --ds-font: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        --ds-text-size: 15px;
        --ds-line-height: 1.55;
    }
    html, body, .stApp, .stApp button, .stApp input, .stApp textarea,
    .stApp select, .stApp table, .stApp [data-testid="stMarkdownContainer"] {
        font-family: var(--ds-font) !important;
    }
    .stApp p, .stApp li, .stApp label,
    .stApp [data-testid="stMarkdownContainer"] p {
        font-size: var(--ds-text-size) !important;
        line-height: var(--ds-line-height) !important;
    }
    .stApp h1 { font-size: 2rem !important; line-height: 1.2 !important; }
    .stApp h2 { font-size: 1.55rem !important; line-height: 1.25 !important; }
    .stApp h3 { font-size: 1.25rem !important; line-height: 1.3 !important; }
    .stApp h4 { font-size: 1.05rem !important; line-height: 1.35 !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        font-family: var(--ds-font) !important;
        font-weight: 650 !important;
        letter-spacing: -0.015em !important;
    }
    .hero-banner-title {
        font-family: var(--ds-font) !important;
        font-size: clamp(1.8rem, 2.7vw, 2.35rem) !important;
        line-height: 1.18 !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    .hero-banner-sub {
        font-family: var(--ds-font) !important;
        font-size: 0.98rem !important;
        line-height: 1.55 !important;
    }
    .section-heading h3 { font-size: 1.22rem !important; }
    .section-heading p { font-size: 0.9rem !important; }
    .section-kicker, .section-label {
        font-family: var(--ds-font) !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
    }
    .sidebar-brand-name, .sidebar-brand-tag, .robot-speech,
    .dataset-card-name, .dataset-card-meta, .dataset-card-badge,
    .stat-header, .stat-value, .stat-sub-inline,
    .panel-title, .panel-row-label, .panel-row-value,
    .quality-row, .insight-row, .system-status {
        font-family: var(--ds-font) !important;
    }
    .sidebar-brand-name { font-size: 1.45rem !important; font-weight: 700 !important; }
    .sidebar-brand-tag { font-size: 0.72rem !important; }
    .robot-speech { font-size: 0.94rem !important; line-height: 1.45 !important; }
    .dataset-card-name, .panel-title {
        font-size: 1rem !important;
        font-weight: 650 !important;
    }
    .dataset-card-meta, .dataset-card-badge, .stat-header,
    .stat-sub-inline, .panel-row-label, .quality-row {
        font-size: 0.82rem !important;
        line-height: 1.45 !important;
    }
    .stat-value { font-size: 1.5rem !important; line-height: 1.2 !important; }
    .panel-row-value, .insight-row { font-size: 0.92rem !important; }
    .stApp .stButton > button,
    .stApp [data-testid="stChatInput"] textarea,
    .stApp [data-testid="stTextInput"] input {
        font-family: var(--ds-font) !important;
        font-size: 0.94rem !important;
        font-weight: 550 !important;
    }
    .stApp [data-testid="stChatMessage"] p,
    .stApp [data-testid="stChatMessage"] li {
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
    }
    .stApp [data-testid="stMetricLabel"] p { font-size: 0.82rem !important; }
    .stApp [data-testid="stMetricValue"] { font-size: 1.65rem !important; }

    /* Keep the message bar and prompt shortcuts together at the bottom of the
       workspace. New messages are processed before this container is drawn. */
    .st-key-chat_composer {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 1000 !important;
        padding: 0.65rem 0 0.55rem !important;
        background: linear-gradient(
            180deg,
            rgba(5, 10, 28, 0.15) 0%,
            rgba(5, 10, 28, 0.96) 20%,
            rgba(5, 10, 28, 0.99) 100%
        ) !important;
        backdrop-filter: blur(12px);
    }
    .st-key-chat_transcript {
        min-height: 42vh;
    }

    /* Keep Streamlit's drag-and-drop instructions, but remove only its upload
       arrow and the custom pointer above the dropzone. */
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] svg,
    .upload-animation::before,
    .upload-animation::after {
        display: none !important;
        content: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# SIDEBAR BRANDING
# ==========================================================

with st.sidebar:
    st.markdown(
        compact_html("""
        <div class="ds-sidebar-brand" aria-label="DataSense AI">
            <div class="ds-logo-stage">
                <div class="ds-logo-glow"></div>

                <div class="ds-orbit ds-orbit-outer" aria-hidden="true">
                    <div class="ds-orbit-icon ds-icon-file">
                        <span class="ds-icon-face ds-counter-outer">
                            <svg viewBox="0 0 24 24" role="img">
                                <path d="M6 2.8h7l5 5V21H6z" />
                                <path d="M13 2.8v5h5M8.7 13h6.6M8.7 16.5h5" />
                            </svg>
                        </span>
                    </div>
                    <div class="ds-orbit-icon ds-icon-chart">
                        <span class="ds-icon-face ds-counter-outer">
                            <svg viewBox="0 0 24 24" role="img">
                                <path d="M4 20V9m5 11V4m5 16v-7m5 7V7" />
                                <path d="M3 20.5h18" />
                            </svg>
                        </span>
                    </div>
                </div>

                <div class="ds-orbit ds-orbit-inner" aria-hidden="true">
                    <div class="ds-orbit-icon ds-icon-database">
                        <span class="ds-icon-face ds-counter-inner">
                            <svg viewBox="0 0 24 24" role="img">
                                <ellipse cx="12" cy="5" rx="7" ry="3" />
                                <path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
                            </svg>
                        </span>
                    </div>
                    <div class="ds-orbit-icon ds-icon-ai">
                        <span class="ds-icon-face ds-counter-inner">
                            <svg viewBox="0 0 24 24" role="img">
                                <path d="M12 2.8c.7 4.7 2.5 6.5 7.2 7.2-4.7.7-6.5 2.5-7.2 7.2-.7-4.7-2.5-6.5-7.2-7.2C9.5 9.3 11.3 7.5 12 2.8Z" />
                                <path d="M18.3 15.5c.3 2 1.2 2.9 3.2 3.2-2 .3-2.9 1.2-3.2 3.2-.3-2-1.2-2.9-3.2-3.2 2-.3 2.9-1.2 3.2-3.2Z" />
                            </svg>
                        </span>
                    </div>
                </div>

                <div class="ds-core">
                    <div class="ds-core-grid"></div>
                    <span>DS</span>
                    <small>AI</small>
                </div>
            </div>
            <div class="sidebar-brand-name">DataSense AI</div>
            <div class="sidebar-brand-tag">Agentic AI Data Analyst</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    if st.session_state.get("nav_active") == "Future Roadmap":
        st.session_state.nav_active = "Data Science Lab"
    elif "nav_active" not in st.session_state or st.session_state.nav_active in ("Home", "Chat"):
        st.session_state.nav_active = "Workspace"

    nav_items = [
        ("Workspace", ":material/home:", None),
        ("Data Quality", ":material/verified:", None),
        ("Insights", ":material/lightbulb:", None),
        ("Learned Rules", ":material/psychology:", None),
        ("Visualisation", ":material/bar_chart:", None),
        ("Blending", ":material/merge:", None),
        ("Data Science Lab", ":material/science:", None),
    ]

    for label, nav_icon, auto_question in nav_items:

        is_active = st.session_state.nav_active == label

        clicked = st.button(
            label,
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            icon=nav_icon,
        )

        if clicked:
            st.session_state.nav_active = label
            st.session_state.pending_question = auto_question
            st.rerun()

    st.markdown(
        '<div class="section-label" style="margin-top:22px;">Recent</div>',
        unsafe_allow_html=True
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    past_questions = [
        m["content"] for m in st.session_state.messages
        if m["role"] == "user"
    ]

    if past_questions:
        for i, q in enumerate(reversed(past_questions[-6:])):
            short_q = q if len(q) <= 30 else q[:30] + "..."
            if st.button(short_q, key=f"history_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()
    else:
        st.caption("No conversations yet")

    st.markdown(
        """
        <div class="system-status">
            <span class="status-dot"></span>
            <div>
                <strong>Local AI online</strong>
                <small>Your data stays on this device</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# STATIC PAGE ROUTER - Blending
# ==========================================================

if st.session_state.nav_active == "Blending":

    st.markdown(BLENDING_CONTENT)
    st.stop()


# ==========================================================
# WORKSPACE HERO
# ==========================================================

# The large product banner belongs to Workspace. Data Science Lab has its own
# model-focused hero and card layout, so repeating this banner there wastes the
# first screen and makes the page feel like Workspace again.
if st.session_state.nav_active != "Data Science Lab":
    hero_text_col, hero_img_col = st.columns([3, 2])

    with hero_text_col:
        st.markdown(
            """
            <div class="hero-banner-title">Turn raw data into <span class="accent">clear decisions.</span></div>
            <div class="hero-banner-sub">AI-powered analysis, visualisation, and business insights in one workspace.</div>
            """,
            unsafe_allow_html=True
        )

    with hero_img_col:
        st.markdown(
            """
            <div class="hero-3d-anchor" aria-hidden="true">
                <div class="data-orbit-3d">
                    <span class="orbit-ring orbit-ring-one"></span>
                    <span class="orbit-ring orbit-ring-two"></span>
                    <span class="orbit-node orbit-node-one"></span>
                    <span class="orbit-node orbit-node-two"></span>
                    <div class="orbit-core">AI</div>
                </div>
                <div class="robot-speech">
                    Drop your dataset below
                    <span>I will turn it into insights.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if HERO_IMAGE_PATH.exists():
            st.image(str(HERO_IMAGE_PATH), use_container_width=True)
        else:
            st.markdown(
                '<div class="hero-fallback">DATASENSE AI</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# FILE UPLOAD
# ==========================================================

if st.session_state.nav_active == "Data Science Lab":
    render_data_science_lab_intro()

    st.markdown(
        """
        <style>
        .st-key-dataset_uploader {
            max-width: 620px;
            margin: 0 auto 0.65rem;
        }
        .st-key-dataset_uploader [data-testid="stFileUploaderDropzone"] {
            min-height: 66px !important;
            padding: 0.55rem 0.85rem !important;
            border: 1px solid rgba(71, 184, 255, 0.42) !important;
            border-radius: 13px !important;
            background:
                linear-gradient(
                    105deg,
                    rgba(10, 48, 104, 0.82),
                    rgba(5, 28, 67, 0.90)
                ) !important;
        }
        .st-key-dataset_uploader
        [data-testid="stFileUploaderDropzoneInstructions"] {
            padding: 0 !important;
        }
        .st-key-dataset_uploader
        [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
            font-size: 0.82rem !important;
            color: #d8edff !important;
        }
        .st-key-dataset_uploader
        [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
            display: none !important;
        }
        .st-key-dataset_uploader button {
            min-height: 34px !important;
            padding: 0.3rem 0.85rem !important;
            border: 1px solid rgba(78, 202, 255, 0.55) !important;
            background: rgba(0, 116, 217, 0.22) !important;
            color: #e8f8ff !important;
        }
        .ds-lab-upload-label {
            max-width: 620px;
            margin: 0 auto 0.42rem;
            color: #7fcfff;
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }
        </style>
        <div class="ds-lab-upload-label">Dataset · CSV or Excel</div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="upload-animation" aria-hidden="true">
            <div class="animated-file"><span>CSV</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# FIX: an empty string "" label (even with label_visibility=
# "collapsed") still triggers Streamlit's "label got an empty
# value" accessibility warning on every rerun - that is what was
# spamming the terminal. A real label + collapsed visibility gets
# the same visual result without the warning.
uploaded_file = st.file_uploader(
    "Upload dataset",
    type=["csv", "xlsx"],
    label_visibility="collapsed",
    key="dataset_uploader",
    on_change=handle_dataset_uploader_change,
)


# The uploader widget is not the dataset's source of truth. Streamlit can
# temporarily return None while widgets are rebuilt during navigation, so keep
# the bytes and metadata in session state and restore them on the next page.
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    uploaded_file_name = uploaded_file.name
    uploaded_file_size = uploaded_file.size
    st.session_state.uploaded_file_bytes = file_bytes
    st.session_state.uploaded_file_name = uploaded_file_name
    st.session_state.uploaded_file_size = uploaded_file_size
elif st.session_state.get("uploaded_file_bytes") is not None:
    file_bytes = st.session_state.uploaded_file_bytes
    uploaded_file_name = st.session_state.uploaded_file_name
    uploaded_file_size = st.session_state.uploaded_file_size
else:
    if st.session_state.nav_active == "Data Science Lab":
        render_data_science_lab(
            None,
            "no-dataset",
            show_intro=False,
        )
    st.stop()


# ==========================================================
# LOAD DATA
# ==========================================================

dataset_id = hashlib.sha256(file_bytes).hexdigest()[:16]

try:
    # Keep an untouched copy plus the active working copy. Cleaning changes
    # only the working copy and survives Streamlit reruns for this dataset.
    if st.session_state.get("active_dataset_id") != dataset_id:
        if uploaded_file_name.lower().endswith(".csv"):
            loaded_df = pd.read_csv(BytesIO(file_bytes))
        else:
            loaded_df = pd.read_excel(BytesIO(file_bytes))

        st.session_state.active_dataset_id = dataset_id
        st.session_state.original_df = loaded_df.copy()
        st.session_state.active_df = loaded_df.copy()
        st.session_state.messages = []
        st.session_state.last_analysis_plan = None
        st.session_state.last_analysis_question = None
        st.session_state.pending_feedback_rule = None
        st.session_state.cleaning_log = []
        for key in list(st.session_state):
            if str(key).startswith("predictive_result_"):
                st.session_state.pop(key, None)

    df = st.session_state.active_df.copy()

except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()


rows, columns = df.shape

duplicate_rows = df.duplicated().sum()

memory = df.memory_usage().sum() / 1024 / 1024

missing_cells = df.isna().sum().sum()

missing_pct = round(missing_cells / (rows * columns) * 100, 2) if rows and columns else 0

file_size_mb = uploaded_file_size / 1024 / 1024

dataset_info = f"""
Rows : {rows}
Columns : {columns}

Column Names:
{", ".join(map(str, df.columns))}

Column Data Types:
{df.dtypes.to_string()}

First 5 Rows

{df.head().to_string(index=False)}
"""

quality_report = profile_data_quality(df)
health_score = calculate_health_score(df, quality_report)
grounded_insights = generate_business_insights(df)
grounded_recommendations = generate_recommendations(grounded_insights)


# ==========================================================
# DATA SCIENCE LAB - PREDICTIVE MODELLING R1
# ==========================================================

if st.session_state.nav_active == "Data Science Lab":
    render_data_science_lab(
        df,
        dataset_id,
        show_intro=False,
    )
    st.stop()


# ==========================================================
# LEARNED RULES PAGE - V2 FEATURE 6
# ==========================================================

if st.session_state.nav_active == "Learned Rules":
    st.markdown("## Learned Rules")
    st.caption(
        "Confirmed corrections for this dataset structure. "
        "DataSense AI checks these rules before planning a similar analysis."
    )

    learned_rules = list_feedback_rules(df)

    if not learned_rules:
        st.info(
            "No corrections have been learned yet. After an incorrect result, "
            "tell DataSense AI what was wrong and which columns it should use."
        )
    else:
        for rule in learned_rules:
            with st.container(border=True):
                st.markdown(f"**Rule {rule['id']}: {rule['instruction']}**")
                details = []
                for label, key in (
                    ("Measure", "measure"),
                    ("Group by", "group_by"),
                    ("Date column", "date_column"),
                    ("Operation", "operation"),
                    ("Time level", "time_granularity"),
                ):
                    if rule.get(key):
                        details.append(f"{label}: {rule[key]}")
                if details:
                    st.caption(" | ".join(details))
                st.caption(f"Learned from: {rule['original_question']}")
                if st.button("Delete rule", key=f"delete_rule_{rule['id']}"):
                    delete_feedback_rule(rule["id"])
                    st.rerun()

    st.stop()


# ==========================================================
# DATA QUALITY PAGE - V2
# ==========================================================

if st.session_state.nav_active == "Data Quality":

    st.markdown("## Data Quality Report")
    st.caption("Every issue and score below is calculated directly from the active dataset.")

    score_col, issue_col, dup_col, outlier_col = st.columns(4)
    score_col.metric("Dataset Health", f"{health_score['Overall']}/100")
    issue_col.metric("Missing Columns", quality_report["missing_columns"])
    dup_col.metric("Duplicate Rows", quality_report["duplicate_rows"])
    outlier_col.metric(
        "Outliers",
        int(quality_report["outliers"]["Outliers"].sum())
        if not quality_report["outliers"].empty else 0,
    )

    st.markdown("### Dataset Health Score")
    for component in ("Completeness", "Consistency", "Duplicates", "Missing Values", "Outliers"):
        left, right = st.columns([5, 1])
        with left:
            st.progress(int(round(health_score[component])))
        with right:
            st.markdown(f"**{component}: {health_score[component]:.1f}**")

    with st.expander("How is the score calculated?"):
        st.write(
            "Overall score = 25% Completeness + 20% Consistency + "
            "20% Duplicate quality + 20% Missing-row quality + 15% Outlier quality. "
            "The score is diagnostic - not a guarantee that the data is suitable for every analysis."
        )

    st.markdown("### Detected Issues")
    issue_tabs = st.tabs([
        "Missing Values", "Duplicates", "Outliers", "Data Types",
        "Constant Columns", "High Cardinality",
    ])

    with issue_tabs[0]:
        if quality_report["missing"].empty:
            st.success("No missing values found.")
        else:
            st.dataframe(quality_report["missing"], use_container_width=True)
    with issue_tabs[1]:
        st.write(f"**{quality_report['duplicate_rows']:,}** duplicate rows detected.")
        if quality_report["duplicate_rows"]:
            st.dataframe(df[df.duplicated(keep=False)].head(100), use_container_width=True)
    with issue_tabs[2]:
        if quality_report["outliers"].empty:
            st.success("No IQR outliers detected in eligible numeric columns.")
        else:
            st.dataframe(quality_report["outliers"], use_container_width=True)
    with issue_tabs[3]:
        if quality_report["incorrect_types"].empty:
            st.success("No likely incorrect data types detected.")
        else:
            st.dataframe(quality_report["incorrect_types"], use_container_width=True)
    with issue_tabs[4]:
        if quality_report["constant_columns"]:
            st.write(quality_report["constant_columns"])
        else:
            st.success("No constant columns detected.")
    with issue_tabs[5]:
        if quality_report["high_cardinality"].empty:
            st.success("No high-cardinality text columns detected.")
        else:
            st.dataframe(quality_report["high_cardinality"], use_container_width=True)

    st.markdown("### Cleaning Assistant")
    if quality_report["missing_columns"]:
        st.info(
            f"I found {quality_report['missing_columns']} columns with missing values. "
            "Choose how DataSense AI should handle each one."
        )
    else:
        st.success("There are no missing values to clean.")

    suggestions = cleaning_suggestions(df, quality_report)
    with st.form("cleaning_form"):
        missing_actions = {}
        for column, options in suggestions.items():
            missing_actions[column] = st.selectbox(
                f"{column} ({int(df[column].isna().sum()):,} missing)",
                options,
                key=f"clean_{dataset_id}_{column}",
            )

        remove_duplicates = st.checkbox(
            f"Remove {quality_report['duplicate_rows']:,} duplicate rows",
            disabled=quality_report["duplicate_rows"] == 0,
        )
        convert_types = st.checkbox(
            f"Convert {len(quality_report['incorrect_types'])} likely incorrect data types",
            disabled=quality_report["incorrect_types"].empty,
        )
        outlier_action = st.selectbox(
            "Outlier treatment",
            ["Ignore", "Cap at IQR bounds", "Remove outlier rows"],
            help="Outliers are not automatically errors. Review them before changing the data.",
        )
        apply_clicked = st.form_submit_button("Apply selected cleaning", type="primary")

    if apply_clicked:
        cleaned, actions = apply_cleaning(
            df,
            missing_actions=missing_actions,
            remove_duplicates=remove_duplicates,
            convert_types=convert_types,
            outlier_action=outlier_action,
            report=quality_report,
        )
        st.session_state.active_df = cleaned
        st.session_state.cleaning_log.extend(actions)
        st.session_state.last_analysis_plan = None
        st.rerun()

    action_col, reset_col = st.columns(2)
    with action_col:
        st.download_button(
            "Download cleaned CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"cleaned_{Path(uploaded_file_name).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with reset_col:
        if st.button("Reset to original data", use_container_width=True):
            st.session_state.active_df = st.session_state.original_df.copy()
            st.session_state.cleaning_log = []
            st.session_state.last_analysis_plan = None
            st.rerun()

    if st.session_state.cleaning_log:
        with st.expander("Cleaning audit log", expanded=True):
            for item in st.session_state.cleaning_log:
                st.write(f"Completed: {item}")

    st.stop()


# ==========================================================
# INSIGHTS & RECOMMENDATIONS PAGE - V2
# ==========================================================

if st.session_state.nav_active == "Insights":
    report_title = f"{Path(uploaded_file_name).stem} Business Insights Report"
    decision_report = generate_decision_report(df, quality_report=quality_report)
    generated_markdown = report_to_markdown(decision_report, report_title)
    chart_specs = generate_report_chart_specs(df)

    # Build-scoped keys prevent an updated Insights engine from reusing stale
    # widget/report state left by an older Streamlit session.
    editor_key = f"insights_report_editor_{APP_BUILD}_{dataset_id}"
    chart_key = f"insights_report_charts_{APP_BUILD}_{dataset_id}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = generated_markdown
    if chart_key not in st.session_state:
        st.session_state[chart_key] = [spec["id"] for spec in chart_specs]

    st.markdown("## Business Insights Report")
    st.caption(
        "Decision-focused findings calculated from the uploaded dataset. "
        f"Edit the narrative, select supporting charts, and download the final report. · Build {APP_BUILD}"
    )

    control_left, control_right = st.columns([4, 1])
    with control_left:
        include_charts = st.checkbox(
            "Show selected charts in report preview",
            value=True,
            key=f"insights_include_charts_{dataset_id}",
        )
    with control_right:
        if st.button("Regenerate report", use_container_width=True, key=f"regenerate_report_{dataset_id}"):
            st.session_state[editor_key] = generated_markdown
            st.rerun()

    preview_tab, edit_tab, charts_tab = st.tabs(["Report preview", "Edit & download", "Charts"])

    with preview_tab:
        st.markdown(st.session_state[editor_key])
        selected_chart_ids = set(st.session_state.get(chart_key, []))
        if include_charts and selected_chart_ids:
            st.markdown("## Supporting Charts")
            for spec in chart_specs:
                if spec["id"] in selected_chart_ids:
                    render_report_chart(spec)

    with edit_tab:
        edited_markdown = st.text_area(
            "Edit report",
            key=editor_key,
            height=720,
            help="The editor uses Markdown. Your changes are reflected in Report preview after the app reruns.",
        )
        html_download = report_to_html(edited_markdown, report_title)
        safe_name = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in Path(uploaded_file_name).stem
        ).strip("_") or "dataset"
        download_left, download_right = st.columns(2)
        with download_left:
            st.download_button(
                "Download Markdown",
                data=edited_markdown.encode("utf-8"),
                file_name=f"{safe_name}_business_insights.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with download_right:
            st.download_button(
                "Download HTML",
                data=html_download.encode("utf-8"),
                file_name=f"{safe_name}_business_insights.html",
                mime="text/html",
                use_container_width=True,
            )
        st.caption("Charts remain interactive in DataSense AI; the downloads contain your edited report narrative.")

    with charts_tab:
        if chart_specs:
            chart_titles = {spec["id"]: spec["title"] for spec in chart_specs}
            st.multiselect(
                "Choose charts for the report preview",
                options=list(chart_titles),
                format_func=lambda chart_id: chart_titles[chart_id],
                key=chart_key,
            )
            selected_chart_ids = set(st.session_state.get(chart_key, []))
            for spec in chart_specs:
                if spec["id"] in selected_chart_ids:
                    render_report_chart(spec)
        else:
            st.info("This dataset does not contain enough supported fields for an automatic report chart.")

    st.info(
        "The report uses dataset calculations only. Possible explanations and recommendations "
        "are clearly separated from facts and should be validated before action."
    )
    st.stop()


# ==========================================================
# VISUALISATION - real drag-and-drop chart builder
# ==========================================================

if st.session_state.nav_active == "Visualisation":

    st.markdown("## Visualisation")

    if not SORTABLES_AVAILABLE:

        st.error(
            "This page needs the `streamlit-sortables` package, which isn't "
            "installed yet. Run this in your terminal (same environment you "
            "use to run Streamlit), then restart the app:\n\n"
            "`pip install streamlit-sortables`"
        )
        st.stop()

    st.caption(
        "Drag fields between the boxes to build a chart - "
        "one field into **Rows**, one into **Values**, "
        "and optionally one into **Color** for a breakdown. "
        "Each tab below is an independent chart."
    )

    dims_all = df.select_dtypes(exclude="number").columns.tolist()
    measures_all = df.select_dtypes(include="number").columns.tolist()

    tab_labels = ["Chart 1", "Chart 2", "Chart 3"]
    tab_ids = ["tab1", "tab2", "tab3"]

    tabs = st.tabs(tab_labels)

    for tab, tab_id in zip(tabs, tab_ids):

        with tab:

            # Keyed by dataset identity + tab, so switching files
            # resets the shelves but each tab keeps its own
            # independent chart setup.
            viz_key = f"viz_shelves_{uploaded_file_name}_{rows}_{columns}_{tab_id}"

            if viz_key not in st.session_state:

                st.session_state[viz_key] = [
                    {"header": "Dimensions", "items": dims_all},
                    {"header": "Measures", "items": measures_all},
                    {"header": "Rows", "items": []},
                    {"header": "Color (optional)", "items": []},
                    {"header": "Values", "items": []},
                ]

            st.session_state[viz_key] = sort_items(
                st.session_state[viz_key],
                multi_containers=True,
                direction="vertical",
                key=f"sortables_{viz_key}"
            )

            shelves = {s["header"]: s["items"] for s in st.session_state[viz_key]}

            rows_shelf = shelves.get("Rows", [])
            color_shelf = shelves.get("Color (optional)", [])
            values_shelf = shelves.get("Values", [])

            st.markdown("<br>", unsafe_allow_html=True)

            ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 1.3])

            with ctrl1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Bar", "Line", "Area", "Pie", "Scatter"],
                    key=f"chart_type_{tab_id}"
                )

                bar_style = st.radio(
                    "Bar style",
                    ["Normal", "3D"],
                    horizontal=True,
                    key=f"bar_style_{tab_id}",
                    disabled=chart_type != "Bar",
                )

            with ctrl2:
                operation = st.selectbox(
                    "Aggregation",
                    ["Sum", "Average", "Count", "Min", "Max", "Median"],
                    key=f"agg_{tab_id}"
                )

            with ctrl3:
                top_n = st.slider(
                    "Top N (0 = all)", 0, 50, 10,
                    key=f"topn_{tab_id}"
                )

            with ctrl4:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                generate = st.button(
                    "Generate chart",
                    key=f"gen_{tab_id}",
                    type="primary",
                    use_container_width=True
                )

            st.markdown("#### Chart preview")

            if generate:

                if not rows_shelf or not values_shelf:

                    st.warning(
                        "Drag a field into **Rows** and one into **Values** first."
                    )

                elif values_shelf[0] not in measures_all:

                    st.warning(
                        f"'{values_shelf[0]}' isn't numeric - drag a field from "
                        "**Measures** into **Values** instead."
                    )

                else:

                    effective_chart_type = (
                        "3D Bar"
                        if chart_type == "Bar" and bar_style == "3D"
                        else chart_type
                    )

                    result = render_manual_chart(
                        df,
                        rows_shelf[0],
                        values_shelf[0],
                        operation,
                        effective_chart_type,
                        color_by=color_shelf[0] if color_shelf else None,
                        top_n=top_n if top_n > 0 else None
                    )

                    st.dataframe(result, use_container_width=True)

            else:

                st.caption("Set up your shelves above, then click Generate.")

    st.stop()


# ==========================================================
# DATASET STATUS CARD
# ==========================================================

st.markdown(
    f"""
    <div class="dataset-card">
        <div class="dataset-card-left">
            <div class="dataset-card-icon">DATA</div>
            <div>
                <div class="dataset-card-name">{uploaded_file_name}</div>
                <div class="dataset-card-meta">{rows:,} rows &nbsp;&middot;&nbsp; {columns} columns &nbsp;&middot;&nbsp; {file_size_mb:.2f} MB</div>
            </div>
        </div>
        <div class="dataset-card-badge">Loaded</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# STAT CARDS
# ==========================================================

stat_defs = [
    ("ROW", "Rows", f"{rows:,}", "Total rows", "#6B4EFF"),
    ("COL", "Columns", f"{columns}", "Total columns", "#3B82F6"),
    ("MB", "Memory Usage", f"{memory:.2f} MB", "Dataset size", "#14B8A6"),
    ("DUP", "Duplicates", f"{duplicate_rows}", "Duplicate rows", "#8B5CF6"),
    ("NA", "Missing Cells", f"{missing_cells} ({missing_pct}%)", "Missing values", "#F43F5E"),
]

stat_cols = st.columns(5)

for col, (icon, label, value, sub, color) in zip(stat_cols, stat_defs):

    with col:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-icon" style="background:{color}26;color:{color};">{icon}</div>
                    <div class="stat-sub-inline">{sub}</div>
                </div>
                <div class="stat-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================================
# OVERVIEW / QUALITY / INSIGHTS PANELS
# ==========================================================

completeness = health_score["Completeness"]
consistency = health_score["Consistency"]
validity = health_score["Overall"]
duplicate_pct = round((duplicate_rows / rows * 100), 2) if rows else 0

panel1, panel2, panel3 = st.columns(3)

with panel1:

    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">Dataset Overview</div>
            <div class="panel-row"><span class="panel-row-label">File Name</span><span class="panel-row-value">{uploaded_file_name}</span></div>
            <div class="panel-row"><span class="panel-row-label">Rows</span><span class="panel-row-value">{rows:,}</span></div>
            <div class="panel-row"><span class="panel-row-label">Columns</span><span class="panel-row-value">{columns}</span></div>
            <div class="panel-row"><span class="panel-row-label">File Size</span><span class="panel-row-value">{file_size_mb:.2f} MB</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Open details", key="view_overview", use_container_width=True):
        st.session_state.show_overview = not st.session_state.get("show_overview", False)

with panel2:

    st.markdown(
        f"""
            <div class="panel-card">
            <div class="panel-title">Dataset Health: {health_score['Overall']}/100</div>
            <div class="quality-row"><span>Completeness</span><span>{completeness}%</span></div>
            <div class="quality-bar"><div class="quality-fill" style="width:{completeness}%;"></div></div>
            <div class="quality-row"><span>Consistency</span><span>{consistency}%</span></div>
            <div class="quality-bar"><div class="quality-fill" style="width:{consistency}%;"></div></div>
            <div class="quality-row"><span>Validity</span><span>{validity}%</span></div>
            <div class="quality-bar"><div class="quality-fill" style="width:{validity}%;"></div></div>
            <div class="panel-row" style="margin-top:10px;"><span class="panel-row-label">Duplicates</span><span class="panel-row-value">{duplicate_rows} ({duplicate_pct}%)</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Open quality report", key="view_quality", use_container_width=True):
        st.session_state.nav_active = "Data Quality"
        st.rerun()

with panel3:
    insight_rows_html = ""

    if grounded_insights:
        for point in grounded_insights[:4]:
            safe_text = html.escape(point["text"])
            insight_rows_html += f'<div class="insight-row">{safe_text}</div>'
    else:
        insight_rows_html = '<div class="panel-row-label">No supported automatic insights yet</div>'

    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">AI Insights</div>
            {insight_rows_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Open insights and recommendations", key="gen_insights", use_container_width=True):
        st.session_state.nav_active = "Insights"
        st.rerun()

if st.session_state.get("show_overview"):
    with st.expander("Full Dataset", expanded=True):
        st.dataframe(df, use_container_width=True)

if st.session_state.get("show_quality"):
    with st.expander("Full Quality Report", expanded=True):
        quality = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Missing Values": df.isna().sum(),
            "Missing %": (df.isna().sum() / rows * 100).round(2)
        })
        st.dataframe(quality, use_container_width=True)


# ==========================================================
# CHAT SECTION
# ==========================================================

st.markdown(
    """
    <div class="section-heading chat-heading">
        <div><span class="section-kicker">AI WORKSPACE</span><h3>Ask DataSense AI</h3></div>
        <p>Continue naturally with follow-up questions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.session_state.pop("pending_question", None)


# ==========================================================
# ANALYSIS MESSAGE RENDERER
# ==========================================================

def render_dataframe_message(message, fallback_key):
    """Render a saved calculation with a persistent Normal/3D chart choice."""
    result_frame = message["content"]
    plan = message.get("plan")

    if plan:
        st.markdown(f"### {plan.get('title', 'Analysis')}")

        if (
            plan.get("analysis_type") == "average_order_value"
            and len(result_frame) == 1
            and "Average Order Value" in result_frame.columns
        ):
            aov = result_frame.iloc[0]["Average Order Value"]
            total_sales = result_frame.iloc[0].get("Total Sales")
            distinct_orders = result_frame.iloc[0].get("Distinct Orders")
            st.metric("Average Order Value", f"{aov:,.2f}")
            st.caption(
                f"Total Sales {total_sales:,.2f} ÷ "
                f"{int(distinct_orders):,} distinct orders"
            )
            return

        display_plan = dict(plan)
        show_chart = bool(display_plan.get("show_chart", False))

        # If a chart was explicitly requested for AOV, plot the calculated AOV
        # column—not its Total Sales numerator.
        if (
            display_plan.get("analysis_type") == "average_order_value"
            and "Average Order Value" in result_frame.columns
        ):
            display_plan["measure"] = "Average Order Value"

        chart_type = str(display_plan.get("chart") or "table").lower()

        if show_chart and chart_type in ("bar", "3d bar", "3d_bar"):
            chart_key = message.get("message_id", fallback_key)
            chart_style = st.radio(
                "Chart style",
                ["Normal", "3D"],
                index=0,
                horizontal=True,
                key=f"chart_style_{chart_key}",
            )
            display_plan["chart"] = "3d_bar" if chart_style == "3D" else "bar"

        # Correlation results are a one-row statistical summary. When the
        # selected presentation is a scatter plot, render the underlying
        # observation-level pair while keeping the compact summary table.
        if show_chart:
            chart_source = message.get("chart_data", result_frame)
            render_chart(chart_source, display_plan)

    st.dataframe(result_frame, use_container_width=True)


# ==========================================================
# CHAT HISTORY
# ==========================================================

chat_transcript = st.container(key="chat_transcript")

with chat_transcript:
    for message_index, message in enumerate(st.session_state.messages):

        with st.chat_message(message["role"], avatar=chat_avatar(message["role"])):

            if isinstance(message["content"], pd.DataFrame):

                render_dataframe_message(message, f"history_{message_index}")

            else:

                st.write(message["content"])


# ==========================================================
# FEEDBACK CONFIRMATION
# ==========================================================

pending_feedback = st.session_state.get("pending_feedback_rule")

if pending_feedback:
    st.markdown("### Save this correction?")
    st.info(pending_feedback["instruction"])

    correction_details = []
    for label, key in (
        ("Measure", "measure"),
        ("Group by", "group_by"),
        ("Date column", "date_column"),
        ("Operation", "operation"),
        ("Time level", "time_granularity"),
        ("Chart", "chart"),
    ):
        if pending_feedback.get(key):
            correction_details.append(f"{label}: {pending_feedback[key]}")

    if correction_details:
        st.caption(" | ".join(correction_details))

    save_col, discard_col = st.columns(2)

    with save_col:
        if st.button("Save learned rule", type="primary", use_container_width=True):
            rule_id = save_feedback_rule(df, pending_feedback)
            st.session_state.pending_feedback_rule = None
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"Correction saved as learned rule {rule_id}. "
                    "I will apply it to similar questions for this dataset structure."
                ),
            })
            st.rerun()

    with discard_col:
        if st.button("Discard", use_container_width=True):
            st.session_state.pending_feedback_rule = None
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Correction discarded. Nothing was saved.",
            })
            st.rerun()


# ==========================================================
# AI COMPOSER - rendered only when no question is being processed. Typed input
# is queued through on_submit, so the next rerun processes and stores the full
# turn before the composer is drawn again at the bottom.
# ==========================================================

def render_chat_composer() -> None:
    composer = st.container(key="chat_composer")

    with composer:
        st.chat_input(
            "Ask a question about your data...",
            key="ai_workspace_question",
            on_submit=queue_chat_question,
        )

        prompts_bar = st.container(key="suggested_prompts_bar")

        with prompts_bar:
            quick_prompts = [
                ("Explain data", "Explain this dataset"),
                ("Find KPIs", "Suggest KPIs"),
                ("Suggest analysis", "Suggest business analyses"),
                ("Data summary", "Statistical Summary"),
            ]

            prompt_cols = st.columns(4)

            for prompt_col, (_label, _q) in zip(prompt_cols, quick_prompts):
                with prompt_col:
                    if st.button(
                        _label,
                        key=f"quick_{_label}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_question = _q
                        st.rerun()


if question is None:
    render_chat_composer()
    st.stop()

# ==========================================================
# INTENT DETECTION
# ==========================================================

previous_plan = st.session_state.get("last_analysis_plan")

# Clear display preferences are saved immediately; ambiguous calculation
# corrections still require confirmation before becoming persistent rules.
if is_feedback_message(question, previous_plan):
    original_question = st.session_state.get("last_analysis_question") or "Previous analysis"

    with st.spinner("Understanding your correction..."):
        proposal = propose_feedback_rule(
            question,
            original_question,
            previous_plan,
            df,
        )

    st.session_state.messages.append({"role": "user", "content": question})

    if proposal.get("valid"):
        if proposal.get("auto_save"):
            rule_id = save_feedback_rule(df, proposal)
            st.session_state.pending_question = original_question
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"Learned preference {rule_id}: {proposal['instruction']} "
                    "I will apply it to similar questions for this dataset and "
                    "have refreshed the previous result below."
                ),
            })
        else:
            st.session_state.pending_feedback_rule = proposal
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "I found a reusable correction. Please review and confirm it "
                    "before I add it to learned memory."
                ),
            })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "I understand that the result was wrong, but I need a specific correction. "
                "Please name the correct measure, date, or grouping column."
            ),
        })

    st.rerun()

follow_up = is_follow_up(question, previous_plan)

detected_intent = detect_intent(question)
chart_requested = detected_intent == "VISUALIZE"
intent = detected_intent

# Short follow-ups such as "Only for 2023" may not contain an explicit
# calculation verb. If an analysis exists in memory, route them back through
# the calculation engine instead of treating them as generic chat.
if follow_up:
    intent = "CALCULATE"

print(f"### CHAT INTENT: {intent} | FOLLOW_UP: {follow_up} ###")

st.session_state.messages.append(
    {
        "role": "user",
        "content": question
    }
)

with st.chat_message("user", avatar=chat_avatar("user")):
    st.write(question)


# ==========================================================
# SYSTEM PROMPT
# ==========================================================

system_prompt = """
You are DataSense AI.

You are an expert Business Data Analyst.

Rules:

1. Never invent columns.

2. Never invent values.

3. Use only dataset information provided.

4. If information cannot be calculated,
clearly say so.

5. Keep answers concise.

6. Think like a Senior Data Analyst.

7. Give business-oriented answers.

8. Use bullet points whenever possible.
"""


# ==========================================================
# TASK SELECTION
# ==========================================================

if intent == "EXPLAIN":

    task = """
Explain the dataset.

Include:

- Dataset Type

- Probable Industry

- Business Purpose

- Important Columns

- 5 Business Questions

Maximum 200 words.
"""

elif intent == "KPI":

    task = """
Suggest the 5-6 most important KPIs for this dataset only.
Do not exceed 6 KPIs total.

For every KPI include

- KPI Name

- Business Value

- Required Columns

- Recommended Chart
"""

elif intent == "ANALYSIS":

    task = """
Suggest the 5-6 most valuable business analyses for THIS dataset
only - not a generic checklist. Only suggest an analysis if the
dataset's actual columns support it (e.g. skip Geographic Analysis
if there is no location column).
Do not exceed 6 analyses total.

For each analysis include:

- Analysis Name

- What It Reveals

- Columns Needed
"""

elif intent == "SUMMARY":

    task = "Statistical Summary"

elif intent == "CALCULATE":

    task = question

elif intent == "VISUALIZE":

    task = question

else:

    task = question

# ==========================================================
# EXECUTION
# ==========================================================

try:

    # ------------------------------------------------------
    # Statistical Summary
    # ------------------------------------------------------

    if intent == "SUMMARY":

        response = statistical_summary(df)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        with st.chat_message("assistant", avatar=chat_avatar("assistant")):
            st.write(response)

    # ------------------------------------------------------
    # Calculation Engine
    # ------------------------------------------------------

    elif intent in ("CALCULATE", "VISUALIZE"):

        learned_rules = relevant_feedback_rules(df, question)
        planner_question, follow_up = contextualize_question(question, previous_plan)
        planner_question = inject_feedback_context(planner_question, learned_rules)
        plans = create_execution_plan(planner_question, df)

        if isinstance(plans, dict):
            plans = [plans]

        # --------------------------------------------------
        # FINAL SAFETY NET - enforce "no grouping dimension
        # unless the question actually asked for one" one last
        # time, right here at the point of execution. Reuses the
        # SAME validation helper as query_planner.py.

        _q_lower = question.lower()

        for _plan in plans:

            _plan.update(merge_follow_up_plan(_plan, previous_plan, question))

            # Workspace calculations return a table by default. Charts are an
            # opt-in presentation and appear only for an explicit chart request.
            _plan["show_chart"] = chart_requested
            if chart_requested and str(_plan.get("chart") or "table").lower() in (
                "table",
                "kpi",
            ):
                _plan["chart"] = "bar"

            if not follow_up and _plan.get("analysis_type") not in ("top_bottom", "pareto"):

                _plan["group_by"] = [
                    _col for _col in _plan.get("group_by", [])
                    if _dimension_mentioned_in_question(str(_col), _q_lower)
                ]

            # Apply confirmed corrections after ordinary planner validation so
            # a learned mapping cannot be stripped out by generic defaults.
            _plan.update(apply_feedback_rules(_plan, learned_rules))

        for i, plan in enumerate(plans, start=1):

            title = plan.get("title", f"Analysis {i}")

            result = calculate(df, plan)

            # The most recently executed plan becomes the context for the next
            # conversational turn (for example: "Now only California").
            st.session_state.last_analysis_plan = dict(plan)
            st.session_state.last_analysis_question = question

            if plan.get("learned_rule_ids"):
                rule_ids = ", ".join(map(str, plan["learned_rule_ids"]))
                st.caption(f"Applied learned rule: {rule_ids}")

            if isinstance(result, pd.DataFrame):

                analysis_message = {
                    "role": "assistant",
                    "content": result,
                    "plan": dict(plan),
                    "message_id": f"{dataset_id}_{len(st.session_state.messages)}_{i}",
                }

                if (
                    plan.get("analysis_type") == "correlation"
                    and str(plan.get("chart") or "").lower() == "scatter"
                ):
                    pair = [plan.get("measure"), plan.get("measure2")]
                    if all(column in df.columns for column in pair):
                        analysis_message["chart_data"] = (
                            df[pair]
                            .apply(pd.to_numeric, errors="coerce")
                            .dropna()
                        )

                st.session_state.messages.append(analysis_message)

                with st.chat_message("assistant", avatar=chat_avatar("assistant")):
                    render_dataframe_message(
                        analysis_message,
                        analysis_message["message_id"],
                    )

            else:

                st.subheader(title)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result
                })

                st.write(result)

    # ------------------------------------------------------
    # Deterministic conversational responses
    # ------------------------------------------------------

    elif intent == "CHAT" and conversational_reply(question) is not None:

        response = conversational_reply(question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
        })

        with st.chat_message("assistant", avatar=chat_avatar("assistant")):
            st.write(response)

    # ------------------------------------------------------
    # LLM Responses (EXPLAIN, KPI, ANALYSIS and other text)
    # ------------------------------------------------------

    else:

        response = ask_llm(

            system_prompt,

            f"""
Dataset Information

{dataset_info}

Task

{task}
"""
        )

        st.session_state.messages.append({

            "role": "assistant",
            "content": response

        })

        with st.chat_message("assistant", avatar=chat_avatar("assistant")):

            st.write(response)

    # Results were stored in session history. Redraw once so every new answer
    # appears above the composer; this prevents the input and prompt row from
    # jumping around the freshly-rendered result.
    st.rerun()

# ==========================================================
# ERROR
# ==========================================================

except Exception as e:

    print(f"ANALYSIS ERROR: {type(e).__name__}: {e}")
    error_message = (
        "I couldn't complete that analysis. Please check the requested "
        f"columns or calculation. Details: {e}"
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": error_message,
    })
    st.rerun()
