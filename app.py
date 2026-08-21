import hashlib
import html
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd

from auth import sign_in_user, sign_out_user, sign_up_user
from usage_tracker import track_feature_open
from account_portal import (
    render_account_menu,
    render_admin_dashboard,
    render_profile_page,
)

from llm_agent import ask_llm
from intent_agent import detect_intent
from pandas_agent import build_calculation_audit, calculate, statistical_summary
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
# Support both database connector versions. Older local projects only expose
# ``load_demo_sales``; newer projects also provide the bundled-sample fallback.
try:
    from database_connector import load_demo_sales_with_fallback
except ImportError:
    from database_connector import load_demo_sales

    def load_demo_sales_with_fallback():
        return load_demo_sales(), {
            "mode": "database",
            "dataset_key": "supabase-demo-sales",
            "label": "Supabase PostgreSQL",
            "display_name": "Supabase · public.demo_sales",
        }

from rag_engine import (
    DEFAULT_EMBEDDING_MODEL,
    KnowledgeBaseError,
    SUPPORTED_KNOWLEDGE_TYPES,
    available_starter_industries,
    build_knowledge_index,
    configured_retrieval_backend,
    format_retrieved_context,
    retrieve_knowledge,
    starter_glossary_document,
)

try:
    from streamlit_sortables import sort_items
    SORTABLES_AVAILABLE = True
except ImportError:
    SORTABLES_AVAILABLE = False

APP_BUILD = "2026.08.21-AUTH-USAGE-PROFILE-ADMIN-AUTHUX-R14"
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
    "active_dataset_source",
    "database_last_refreshed_at",
)


def clear_active_dataset_state() -> None:
    """Clear the active dataset and every result derived from it."""
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


def handle_dataset_uploader_change() -> None:
    """Persist a new upload, or clear dataset state when its X is clicked."""
    current_file = st.session_state.get("dataset_uploader")

    if current_file is not None:
        st.session_state.uploaded_file_bytes = current_file.getvalue()
        st.session_state.uploaded_file_name = current_file.name
        st.session_state.uploaded_file_size = current_file.size
        return

    clear_active_dataset_state()


def handle_dataset_source_change() -> None:
    """Reset derived results when the user switches between file and SQL."""
    st.session_state.pop("dataset_uploader", None)
    clear_active_dataset_state()


def dataframe_dataset_id(df: pd.DataFrame, source: str) -> str:
    """Create a stable ID that changes when database values or schema change."""
    schema = "|".join(f"{column}:{dtype}" for column, dtype in df.dtypes.items())
    values = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    digest = hashlib.sha256(schema.encode("utf-8") + values).hexdigest()[:16]
    return f"{source}-{digest}"


def activate_dataset(loaded_df: pd.DataFrame, dataset_id: str, source: str) -> None:
    """Make a newly loaded file or database snapshot the active dataset."""
    if st.session_state.get("active_dataset_id") == dataset_id:
        return

    st.session_state.active_dataset_id = dataset_id
    st.session_state.active_dataset_source = source
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


def compact_html(markup: str) -> str:
    """Keep Streamlit Markdown from breaking nested raw HTML at blank lines."""
    return "".join(line.strip() for line in markup.splitlines())


# ==========================================================
# AUTHENTICATION
# ==========================================================

def _remember_authenticated_user(user) -> None:
    """Keep only the small identity fields DataSense needs in session state."""
    metadata = getattr(user, "user_metadata", None) or {}
    st.session_state.auth_user = {
        "id": str(getattr(user, "id", "")),
        "email": str(getattr(user, "email", "") or ""),
        "full_name": str(metadata.get("full_name") or "").strip(),
    }


def _set_auth_mode(mode: str) -> None:
    """Switch the unauthenticated experience between guided auth steps."""
    allowed = {"welcome", "signup", "verify", "login"}
    st.session_state.auth_mode = mode if mode in allowed else "welcome"


def _friendly_auth_error(exc: Exception) -> str:
    """Translate common Supabase auth failures into useful product guidance."""
    message = str(exc or "").strip()
    lowered = message.lower()

    if "email not confirmed" in lowered or "email_not_confirmed" in lowered:
        return (
            "Your account exists, but your email is not confirmed yet. "
            "Open the verification email from DataSense, confirm it, then sign in again."
        )
    if (
        "invalid login credentials" in lowered
        or "invalid_credentials" in lowered
        or "invalid password" in lowered
    ):
        return (
            "That email and password do not match. Check the credentials you used "
            "when creating the account and try again."
        )
    if "user already registered" in lowered or "already registered" in lowered:
        return (
            "An account already exists for this email. Choose Log in and use the "
            "password you created for that account."
        )
    if "rate limit" in lowered or "over_email_send_rate_limit" in lowered:
        return (
            "Supabase has temporarily limited authentication emails. Wait a little "
            "before requesting another email, then try again."
        )

    return (
        "DataSense could not complete that authentication step. Check your details "
        "and try again. If you just created the account, make sure the verification "
        "email has been confirmed first."
    )


def _render_auth_showcase() -> None:
    """Render the branded DataSense D-to-product portal animation."""
    st.markdown(
        compact_html(
            """
            <div class="ds-auth-left">
              <div class="ds-auth-left-label">DATASENSE AI · QUICK TOUR</div>
              <div class="ds-tour-shell ds-portal-tour" aria-hidden="true">
                <div class="ds-tour-topbar">
                  <div class="ds-tour-dots"><i></i><i></i><i></i></div>
                  <div class="ds-tour-title">DataSense AI</div>
                </div>
                <div class="ds-tour-progress"><span></span></div>

                <div class="ds-brand-intro">
                  <div class="ds-brand-glow"></div>
                  <div class="ds-brand-stage">
                    <div class="ds-brand-d">D</div>
                    <div class="ds-brand-rest">
                      <span>A</span><span>T</span><span>A</span><span>S</span><span>E</span><span>N</span><span>S</span><span>E</span>
                    </div>
                    <div class="ds-brand-tagline">Your AI data copilot</div>
                  </div>
                </div>

                <div class="ds-portal-reveal">
                  <div class="ds-portal-frame">
                    <div class="ds-portal-door"><span>D</span></div>
                  </div>
                </div>

                <div class="ds-product-demo">
                  <div class="ds-product-sidebar">
                    <div class="ds-product-brand">DS</div>
                    <div class="ds-product-nav active">Workspace</div>
                    <div class="ds-product-nav">Knowledge Base</div>
                    <div class="ds-product-nav">Data Quality</div>
                    <div class="ds-product-nav">Insights</div>
                    <div class="ds-product-nav">Visualisation</div>
                  </div>

                  <div class="ds-product-main">
                    <div class="ds-product-scene ds-scene-upload">
                      <div class="ds-product-eyebrow">WORKSPACE</div>
                      <div class="ds-product-heading">Bring your data into focus.</div>
                      <div class="ds-product-upload">
                        <div class="ds-product-upload-icon">⇪</div>
                        <div><b>sales_q2.csv</b><small>Uploading to DataSense…</small></div>
                        <div class="ds-product-upload-progress"><span></span></div>
                      </div>
                      <div class="ds-product-file-ready">Loaded · 19,324 rows</div>
                    </div>

                    <div class="ds-product-scene ds-scene-question">
                      <div class="ds-product-eyebrow">ASK DATASENSE</div>
                      <div class="ds-product-heading">Ask in plain English.</div>
                      <div class="ds-product-question">Show sales by region</div>
                      <div class="ds-product-answer">Building the analysis plan…</div>
                    </div>

                    <div class="ds-product-scene ds-scene-chart">
                      <div class="ds-product-eyebrow">VISUALISATION</div>
                      <div class="ds-product-heading">Sales by region</div>
                      <div class="ds-product-chart">
                        <div class="ds-product-bar" style="height:45%"><span>North</span></div>
                        <div class="ds-product-bar" style="height:66%"><span>South</span></div>
                        <div class="ds-product-bar" style="height:58%"><span>East</span></div>
                        <div class="ds-product-bar highlight" style="height:88%"><span>West</span></div>
                      </div>
                      <div class="ds-product-caption">West leads overall sales performance.</div>
                    </div>

                    <div class="ds-product-scene ds-scene-insights">
                      <div class="ds-product-eyebrow">INSIGHTS</div>
                      <div class="ds-product-heading">Generate insights.</div>
                      <div class="ds-product-generate">Generate insights</div>
                      <div class="ds-product-insights">
                        <div><b>Growth driver</b><small>West delivers the strongest revenue uplift.</small></div>
                        <div><b>Risk signal</b><small>South margin is below the overall average.</small></div>
                        <div><b>Recommended next step</b><small>Review regional mix and discounting.</small></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div class="ds-tour-control">❚❚</div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _render_auth_assistant(mode: str) -> None:
    """Render a professional humanoid DataSense assistant on the auth side."""
    messages = {
        "welcome": "Hi, I’m DataSense AI. Ready to go inside?",
        "signup": "Create your account and I’ll guide you through email confirmation next.",
        "verify": "Check your inbox, confirm your email, then come back and sign in.",
        "login": "Welcome back. Sign in and let’s explore your data.",
    }
    safe = html.escape(messages.get(mode, messages["welcome"]))
    st.markdown(
        compact_html(
            f"""
            <div class="ds-assistant-row">
              <div class="ds-pro-robot" aria-hidden="true">
                <div class="pro-aura"></div>
                <div class="pro-head">
                  <div class="pro-faceplate">
                    <i class="pro-eye e1"></i><i class="pro-eye e2"></i>
                    <i class="pro-mouth"></i>
                  </div>
                  <div class="pro-temple t1"></div><div class="pro-temple t2"></div>
                </div>
                <div class="pro-neck"></div>
                <div class="pro-shoulders"><i></i><i></i></div>
                <div class="pro-torso">
                  <div class="pro-core"></div>
                  <b>DS</b><small>AI</small>
                </div>
                <div class="pro-arm arm-left"></div><div class="pro-arm arm-right"></div>
              </div>
              <div class="ds-speech">{safe}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

def render_auth_screen() -> None:
    """Render the compact, above-the-fold DataSense authentication flow."""
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "welcome"

    mode = str(st.session_state.get("auth_mode") or "welcome")
    if mode not in {"welcome", "signup", "verify", "login"}:
        mode = "welcome"
        st.session_state.auth_mode = mode

    if mode != "login":
        st.session_state.pop("auth_sign_in_password", None)
    if mode != "signup":
        st.session_state.pop("auth_sign_up_password", None)
        st.session_state.pop("auth_sign_up_confirm_password", None)

    notice = st.session_state.pop("auth_notice", None)
    if notice == "signed_out":
        try:
            st.toast("Signed out successfully.", icon="✅")
        except Exception:
            pass

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: .45rem !important;
            padding-bottom: .45rem !important;
            max-width: 1500px !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        .ds-auth-left { padding-top: .25rem; }
        .ds-auth-left-label {
            color:#91dcff; font-size:.72rem; font-weight:800; letter-spacing:.16em; margin:0 0 .55rem .15rem;
        }
        .ds-tour-shell {
            position:relative; height:600px; overflow:hidden; border-radius:28px;
            background:linear-gradient(150deg,rgba(5,10,28,.98),rgba(10,14,38,.96));
            border:1px solid rgba(130,145,220,.16);
            box-shadow:0 26px 60px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.03);
        }
        .ds-tour-topbar {
            height:48px; display:flex; align-items:center; gap:8px; padding:0 16px;
            background:linear-gradient(180deg,rgba(22,31,67,.98),rgba(12,18,44,.96));
            border-bottom:1px solid rgba(126,143,210,.13);
        }
        .ds-tour-dots { display:flex; gap:6px; }
        .ds-tour-dots i { display:block; width:9px; height:9px; border-radius:50%; background:#f87171; }
        .ds-tour-dots i:nth-child(2){background:#fbbf24}.ds-tour-dots i:nth-child(3){background:#34d399}
        .ds-tour-title { color:rgba(238,243,255,.9); font-size:.8rem; font-weight:700; margin-left:5px; }
        .ds-tour-status { margin-left:auto; color:#d8dcff; background:rgba(111,97,255,.16); border:1px solid rgba(139,127,255,.2); border-radius:999px; padding:.28rem .58rem; font-size:.69rem; font-weight:750; }
        .ds-tour-progress { height:3px; background:rgba(255,255,255,.05); }
        .ds-tour-progress span { display:block; height:100%; width:100%; transform-origin:left center; background:linear-gradient(90deg,#39d5ef,#735fff); animation:tourProgress 18s linear infinite; }
        .ds-tour-screen { position:absolute; left:0; right:0; top:51px; bottom:0; display:flex; opacity:0; pointer-events:none; }
        .tour-1{animation:tourOne 10s infinite}.tour-2{animation:tourTwo 10s infinite}.tour-3{animation:tourThree 10s infinite}.tour-4{animation:tourFour 10s infinite}
        .mock-sidebar { width:160px; padding:16px 12px; border-right:1px solid rgba(122,139,204,.12); background:rgba(7,14,34,.72); }
        .mock-brand { width:38px; height:38px; display:flex; align-items:center; justify-content:center; border-radius:13px; background:linear-gradient(145deg,#6d5cff,#2e82d5); color:#fff; font-weight:850; margin-bottom:20px; }
        .mock-nav { padding:9px 10px; margin-bottom:6px; border-radius:10px; color:rgba(206,216,242,.62); font-size:.75rem; }
        .mock-nav.active { background:rgba(103,91,255,.16); color:#eef2ff; }
        .mock-main { position:relative; flex:1; padding:30px 34px; overflow:hidden; background:radial-gradient(circle at 75% 10%,rgba(78,74,210,.11),transparent 30%); }
        .mock-eyebrow { color:#8bdcff; font-size:.68rem; font-weight:800; letter-spacing:.14em; }
        .mock-heading { color:#f7f9ff; font-size:1.7rem; font-weight:780; letter-spacing:-.035em; margin-top:.45rem; }
        .mock-heading.small { font-size:1.45rem; }
        .mock-sub { color:rgba(216,225,245,.64); font-size:.9rem; margin-top:.4rem; }
        .mock-upload { margin-top:34px; max-width:560px; height:170px; border:1px dashed rgba(126,220,255,.34); border-radius:24px; display:flex; align-items:center; justify-content:center; gap:18px; background:rgba(15,29,67,.62); color:#eef7ff; }
        .mock-upload-icon { width:58px; height:58px; display:flex; align-items:center; justify-content:center; border-radius:18px; background:rgba(35,68,127,.7); color:#9ef3ff; font-size:1.5rem; }
        .mock-upload b,.mock-upload small{display:block}.mock-upload small{color:rgba(205,217,242,.6);margin-top:5px;font-size:.76rem}
        .mock-file-row { display:flex; justify-content:space-between; align-items:center; padding:13px 15px; border-radius:15px; background:rgba(17,30,70,.72); border:1px solid rgba(128,145,214,.12); color:#f6f9ff; }
        .mock-file-row span { color:#8fdfef; font-size:.75rem; }
        .mock-kpis { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:16px; }
        .mock-kpis>div { padding:15px; border-radius:16px; background:rgba(18,31,72,.72); border:1px solid rgba(128,145,214,.1); }
        .mock-kpis small,.mock-kpis b{display:block}.mock-kpis small{color:rgba(204,216,241,.58);font-size:.72rem}.mock-kpis b{color:#fff;font-size:1.1rem;margin-top:7px}
        .mock-chat-label { margin-top:24px; color:#9fdff5; font-size:.76rem; font-weight:700; }
        .mock-question { margin-top:9px; padding:13px 15px; border-radius:15px; background:rgba(103,91,255,.14); color:#eef2ff; width:66%; font-size:.92rem; }
        .mock-chart { display:flex; align-items:flex-end; gap:22px; height:285px; padding:28px 22px 24px; margin-top:18px; border-radius:22px; background:rgba(16,29,68,.66); border:1px solid rgba(126,145,216,.12); }
        .mock-chart .bar { position:relative; flex:1; min-width:40px; max-width:90px; border-radius:15px 15px 6px 6px; background:linear-gradient(180deg,#7fe2ff,#4d70ec); box-shadow:0 9px 20px rgba(62,99,226,.18); }
        .mock-chart .bar.emphasis { background:linear-gradient(180deg,#9cf3ff,#6557ff); }
        .mock-chart .bar span { position:absolute; bottom:-24px; left:50%; transform:translateX(-50%); color:rgba(214,224,246,.65); font-size:.7rem; white-space:nowrap; }
        .mock-caption { margin-top:26px; color:rgba(224,232,248,.72); font-size:.85rem; }
        .mock-insight-actions { margin-top:18px; }
        .mock-generate { display:inline-flex; padding:10px 15px; border-radius:12px; background:linear-gradient(90deg,#725dff,#5a43e9); color:#fff; font-size:.82rem; font-weight:700; box-shadow:0 9px 22px rgba(96,76,240,.24); }
        .mock-insights { display:grid; gap:12px; margin-top:20px; max-width:640px; }
        .mock-insights>div { padding:16px 17px; border-radius:17px; background:rgba(17,30,70,.72); border:1px solid rgba(128,145,214,.11); }
        .mock-insights b,.mock-insights small{display:block}.mock-insights b{color:#f7f9ff;font-size:.9rem}.mock-insights small{color:rgba(205,217,242,.64);font-size:.77rem;margin-top:4px;line-height:1.45}
        .ds-tour-control { position:absolute; right:16px; bottom:14px; width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:rgba(230,235,246,.92); color:#0b1325; font-size:.72rem; font-weight:900; box-shadow:0 8px 18px rgba(0,0,0,.18); }

        .ds-assistant-row {
            display:flex; align-items:center; justify-content:center; gap:18px;
            margin:18px auto 28px; max-width:420px;
        }
        .ds-pro-robot {
            position:relative; width:132px; min-width:132px; height:182px;
            filter:drop-shadow(0 18px 30px rgba(0,0,0,.28)); animation:miniFloat 5.2s ease-in-out infinite;
        }
        .pro-aura {
            position:absolute; left:50%; top:28px; width:108px; height:108px; transform:translateX(-50%);
            border-radius:50%; background:radial-gradient(circle,rgba(77,202,255,.16),rgba(111,97,255,.05) 55%,transparent 72%);
            filter:blur(3px);
        }
        .pro-head {
            position:absolute; left:50%; top:4px; width:84px; height:58px; transform:translateX(-50%);
            border-radius:15px 15px 12px 12px;
            background:linear-gradient(145deg,#dce8fb 0%,#91a9df 38%,#566abe 100%);
            border:1.5px solid rgba(255,255,255,.40);
            box-shadow:inset 0 -9px 16px rgba(22,37,92,.24), inset 0 5px 9px rgba(255,255,255,.18);
        }
        .pro-faceplate {
            position:absolute; left:9px; right:9px; top:12px; height:34px; border-radius:10px;
            background:linear-gradient(180deg,rgba(5,15,34,.98),rgba(10,27,58,.95));
            border:1px solid rgba(115,220,255,.10);
        }
        .pro-eye { position:absolute; top:12px; width:9px; height:5px; border-radius:999px; background:#8feaff; box-shadow:0 0 10px rgba(120,228,255,.7); }
        .pro-eye.e1 { left:17px; } .pro-eye.e2 { right:17px; }
        .pro-mouth { position:absolute; left:50%; top:23px; width:22px; height:6px; transform:translateX(-50%); border-bottom:1.5px solid rgba(212,244,255,.78); border-radius:0 0 12px 12px; }
        .pro-temple { position:absolute; top:20px; width:7px; height:18px; border-radius:4px; background:linear-gradient(#6b83d9,#3c51a4); }
        .pro-temple.t1 { left:-5px; } .pro-temple.t2 { right:-5px; }
        .pro-neck {
            position:absolute; left:50%; top:58px; width:24px; height:18px; transform:translateX(-50%);
            border-radius:4px; background:linear-gradient(#6076c8,#3d519b);
        }
        .pro-shoulders {
            position:absolute; left:50%; top:72px; width:124px; height:36px; transform:translateX(-50%);
            border-radius:30px 30px 12px 12px; background:linear-gradient(145deg,#647ce0,#2948a7);
            border:1px solid rgba(190,211,255,.18);
        }
        .pro-shoulders i { position:absolute; top:11px; width:26px; height:9px; border-radius:999px; background:rgba(117,224,255,.22); }
        .pro-shoulders i:first-child { left:13px; } .pro-shoulders i:last-child { right:13px; }
        .pro-torso {
            position:absolute; left:50%; top:83px; width:82px; height:88px; transform:translateX(-50%);
            border-radius:12px 12px 22px 22px; background:linear-gradient(160deg,#4f67cb,#253f92 55%,#0d728a);
            border:1.5px solid rgba(190,211,255,.20); display:flex; align-items:center; justify-content:center; gap:4px;
            box-shadow:inset 0 -18px 26px rgba(4,18,71,.28);
        }
        .pro-core {
            position:absolute; top:15px; left:50%; width:34px; height:5px; transform:translateX(-50%); border-radius:999px;
            background:linear-gradient(90deg,#6d5cff,#8feaff); box-shadow:0 0 12px rgba(126,219,255,.45);
        }
        .pro-torso b { color:white; font-size:1.05rem; letter-spacing:.04em; transform:translateY(9px); }
        .pro-torso small { color:#a9f5ff; font-size:.55rem; font-weight:800; transform:translateY(13px); }
        .pro-arm {
            position:absolute; top:88px; width:18px; height:73px; border-radius:11px 11px 14px 14px;
            background:linear-gradient(180deg,#4c64bd,#1f3375); border:1px solid rgba(190,211,255,.12);
        }
        .arm-left { left:4px; transform:rotate(4deg); } .arm-right { right:4px; transform:rotate(-4deg); }
        .ds-speech {
            position:relative; flex:1; padding:14px 16px; border-radius:18px 18px 18px 9px;
            background:rgba(12,21,51,.78); border:1px solid rgba(130,146,217,.14);
            color:rgba(238,243,255,.9); line-height:1.48; font-size:.9rem;
        }
        .ds-speech:before { content:""; position:absolute; left:-8px; bottom:18px; width:16px; height:16px; background:rgba(12,21,51,.78); transform:rotate(45deg); border-left:1px solid rgba(130,146,217,.12); border-bottom:1px solid rgba(130,146,217,.12); }

        .ds-auth-label { color:#8bdcff; font-size:.7rem; font-weight:800; letter-spacing:.14em; margin-bottom:.65rem; }
        .ds-auth-verify-banner { margin:.35rem 0 .75rem; padding:.85rem .9rem .9rem 1rem; border-left:3px solid #6f61ff; border-radius:0 13px 13px 0; background:rgba(111,97,255,.08); color:rgba(228,235,250,.82); line-height:1.5; font-size:.88rem; }
        .ds-auth-steps-inline { display:flex; gap:.45rem; flex-wrap:wrap; margin:.6rem 0 .8rem; }
        .ds-auth-steps-inline span { padding:.38rem .55rem; border-radius:999px; background:rgba(11,20,49,.72); border:1px solid rgba(131,148,221,.14); color:rgba(223,231,248,.78); font-size:.7rem; font-weight:620; }
        .ds-auth-note { margin-top:.6rem; color:rgba(203,212,234,.58); font-size:.8rem; line-height:1.45; }
        .stButton>button[kind="primary"] { min-height:3rem !important; font-weight:700 !important; }
        .stButton>button { min-height:2.9rem !important; }
        .stTextInput>div>div>input { min-height:2.55rem !important; }
        div[data-testid="stForm"] { border:0 !important; padding:0 !important; }

        @keyframes tourProgress { from{transform:scaleX(0)}to{transform:scaleX(1)} }
        @keyframes tourOne { 0%,23%{opacity:1}25%,100%{opacity:0} }
        @keyframes tourTwo { 0%,24%{opacity:0}26%,48%{opacity:1}50%,100%{opacity:0} }
        @keyframes tourThree { 0%,49%{opacity:0}51%,73%{opacity:1}75%,100%{opacity:0} }
        @keyframes tourFour { 0%,74%{opacity:0}76%,98%{opacity:1}100%{opacity:0} }
        @keyframes miniFloat { 0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)} }

        @media (max-height: 820px) and (min-width: 901px) {
          .ds-tour-shell { height:540px; }
          .ds-assistant-row { margin:8px 0 16px; }
          .ds-pro-robot { transform:scale(.92); transform-origin:center; }
        }
        @media (max-width: 1100px) {
          .ds-tour-shell { height:540px; }
          .mock-sidebar { width:135px; }
          .mock-main { padding:24px 26px; }
        }
        @media (max-width: 900px) {
          .ds-tour-shell { height:500px; }
          .mock-sidebar { width:118px; }
          .ds-assistant-row { margin-top:8px; }
        }
        
        /* ---------- R12 DataSense D-to-product portal ---------- */
        .ds-portal-tour { height: 570px; }
        .ds-brand-intro, .ds-portal-reveal, .ds-product-demo {
            position:absolute; left:0; right:0; top:52px; bottom:0;
        }
        .ds-brand-intro {
            z-index:8; display:flex; align-items:center; justify-content:center;
            background:
                radial-gradient(circle at 50% 45%, rgba(75,145,255,.15), transparent 26%),
                radial-gradient(circle at 50% 50%, rgba(124,92,255,.13), transparent 38%);
            animation:dsBrandIntro 18s infinite;
        }
        .ds-brand-glow {
            position:absolute; width:330px; height:330px; border-radius:50%;
            background:radial-gradient(circle, rgba(57,215,239,.16), rgba(124,92,255,.12) 38%, transparent 70%);
            filter:blur(12px); animation:dsBrandGlow 2.2s ease-in-out infinite;
        }
        .ds-brand-stage { position:relative; z-index:2; display:flex; align-items:center; justify-content:center; min-width:540px; height:190px; }
        .ds-brand-d {
            position:relative; z-index:3; color:#f8fbff; font-family:"Space Grotesk","Inter",sans-serif;
            font-size:150px; font-weight:800; line-height:1; letter-spacing:-.09em;
            text-shadow:0 0 25px rgba(101,195,255,.35), 0 0 70px rgba(96,70,230,.24);
            animation:dsLetterD 18s infinite;
        }
        .ds-brand-rest { display:flex; align-items:center; margin-left:8px; overflow:hidden; }
        .ds-brand-rest span {
            display:inline-block; color:#edf5ff; font-family:"Space Grotesk","Inter",sans-serif;
            font-size:72px; font-weight:760; letter-spacing:-.05em; opacity:0; transform:translateX(-30px) scale(.92);
            animation:dsLetterRest 18s infinite;
        }
        .ds-brand-rest span:nth-child(1){animation-delay:.05s}.ds-brand-rest span:nth-child(2){animation-delay:.10s}
        .ds-brand-rest span:nth-child(3){animation-delay:.15s}.ds-brand-rest span:nth-child(4){animation-delay:.20s}
        .ds-brand-rest span:nth-child(5){animation-delay:.25s}.ds-brand-rest span:nth-child(6){animation-delay:.30s}
        .ds-brand-rest span:nth-child(7){animation-delay:.35s}.ds-brand-rest span:nth-child(8){animation-delay:.40s}
        .ds-brand-tagline {
            position:absolute; top:150px; left:50%; transform:translateX(-50%); white-space:nowrap;
            color:rgba(195,223,245,.68); font-size:.78rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase;
            opacity:0; animation:dsTagline 18s infinite;
        }
        .ds-portal-reveal {
            z-index:9; display:flex; align-items:center; justify-content:center; pointer-events:none;
            opacity:0; animation:dsPortalLayer 18s infinite;
        }
        .ds-portal-frame { position:relative; width:210px; height:300px; perspective:900px; }
        .ds-portal-door {
            position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
            border-radius:40px 40px 34px 34px; transform-origin:left center;
            background:linear-gradient(150deg, rgba(75,127,255,.96), rgba(91,63,222,.95) 58%, rgba(20,162,198,.90));
            border:1px solid rgba(204,226,255,.55); box-shadow:0 0 42px rgba(74,112,255,.30), inset 0 1px 0 rgba(255,255,255,.22);
            animation:dsDoorOpen 18s infinite;
        }
        .ds-portal-door span { color:white; font-family:"Space Grotesk","Inter",sans-serif; font-size:126px; font-weight:820; letter-spacing:-.10em; transform:translateX(-5px); }
        .ds-product-demo {
            z-index:5; display:flex; opacity:0; transform:scale(.92); transform-origin:center;
            background:linear-gradient(145deg, rgba(5,10,29,.98), rgba(8,14,36,.98));
            animation:dsProductReveal 18s infinite;
        }
        .ds-product-sidebar {
            width:155px; flex:0 0 155px; padding:22px 14px; background:rgba(6,12,30,.94); border-right:1px solid rgba(133,149,210,.11);
        }
        .ds-product-brand {
            display:grid; place-items:center; width:42px; height:42px; margin:0 auto 24px; border-radius:14px;
            color:#fff; font-weight:800; background:linear-gradient(145deg,#7c5cff,#2478c8); box-shadow:0 10px 22px rgba(78,66,220,.22);
        }
        .ds-product-nav { padding:9px 10px; margin:5px 0; border-radius:9px; color:rgba(181,194,222,.64); font-size:.69rem; font-weight:650; }
        .ds-product-nav.active { color:#fff; background:rgba(124,92,255,.15); border:1px solid rgba(124,92,255,.20); }
        .ds-product-main { position:relative; flex:1; overflow:hidden; padding:28px 34px; }
        .ds-product-scene { position:absolute; inset:28px 34px; opacity:0; transform:translateY(8px); }
        .ds-scene-upload { animation:dsUploadScene 18s infinite; }
        .ds-scene-question { animation:dsQuestionScene 18s infinite; }
        .ds-scene-chart { animation:dsChartScene 18s infinite; }
        .ds-scene-insights { animation:dsInsightsScene 18s infinite; }
        .ds-product-eyebrow { color:#91dcff; font-size:.68rem; font-weight:800; letter-spacing:.16em; }
        .ds-product-heading { margin-top:.6rem; color:#f5f8ff; font-size:1.45rem; font-weight:760; letter-spacing:-.025em; }
        .ds-product-upload {
            position:relative; display:flex; align-items:center; gap:14px; margin-top:1.25rem; padding:18px; border-radius:18px;
            background:rgba(15,27,64,.78); border:1px dashed rgba(139,220,255,.32); max-width:520px;
        }
        .ds-product-upload-icon { display:grid; place-items:center; width:52px; height:52px; border-radius:16px; color:#a6f5ff; background:rgba(34,75,145,.45); font-size:1.35rem; }
        .ds-product-upload b { display:block; color:#fff; }.ds-product-upload small { display:block; margin-top:3px; color:rgba(199,212,238,.63); }
        .ds-product-upload-progress { position:absolute; left:18px; right:18px; bottom:8px; height:3px; overflow:hidden; background:rgba(255,255,255,.06); border-radius:99px; }
        .ds-product-upload-progress span { display:block; width:100%; height:100%; background:linear-gradient(90deg,#22d3ee,#7c5cff); animation:dsUploadProgress 18s infinite; transform-origin:left; }
        .ds-product-file-ready { margin-top:.85rem; color:#9ae6c8; font-size:.78rem; font-weight:650; }
        .ds-product-question { margin-top:1.2rem; width:fit-content; max-width:78%; padding:12px 15px; border-radius:16px 16px 16px 6px; background:rgba(105,88,255,.16); color:#fff; font-weight:650; }
        .ds-product-answer { margin:12px 0 0 auto; width:fit-content; padding:11px 14px; border-radius:16px 16px 6px 16px; background:rgba(17,31,72,.86); color:rgba(218,228,248,.77); font-size:.82rem; }
        .ds-product-chart { display:flex; align-items:flex-end; gap:14px; height:215px; margin-top:1rem; padding:10px 12px 30px; border-radius:18px; background:rgba(14,25,59,.58); }
        .ds-product-bar { position:relative; flex:1; border-radius:14px 14px 5px 5px; background:linear-gradient(180deg,rgba(95,159,255,.78),rgba(73,87,209,.78)); box-shadow:0 10px 20px rgba(45,74,190,.15); }
        .ds-product-bar.highlight { background:linear-gradient(180deg,#71e1f5,#6e61ff); box-shadow:0 12px 28px rgba(75,145,255,.25); }
        .ds-product-bar span { position:absolute; left:50%; bottom:-23px; transform:translateX(-50%); color:rgba(201,212,236,.63); font-size:.68rem; }
        .ds-product-caption { margin-top:.65rem; color:rgba(214,224,244,.72); font-size:.82rem; }
        .ds-product-generate { margin-top:1rem; display:inline-flex; padding:9px 14px; border-radius:11px; background:linear-gradient(110deg,#7c5cff,#5b3fde); color:#fff; font-size:.78rem; font-weight:720; box-shadow:0 10px 24px rgba(91,63,222,.22); }
        .ds-product-insights { display:grid; gap:8px; margin-top:.85rem; max-width:560px; }
        .ds-product-insights > div { padding:10px 12px; border-radius:13px; background:rgba(17,30,70,.72); border:1px solid rgba(127,146,219,.10); }
        .ds-product-insights b { display:block; color:#f6f8ff; font-size:.78rem; }.ds-product-insights small { display:block; margin-top:2px; color:rgba(199,211,236,.62); font-size:.69rem; }

        /* R14: fast branded flash, then move immediately into the product tour. */
        @keyframes dsBrandIntro {
            0%, 7% { opacity:1; visibility:visible; }
            9%, 100% { opacity:0; visibility:hidden; }
        }
        @keyframes dsBrandGlow {
            0%,100% { transform:scale(.92); opacity:.52; }
            50% { transform:scale(1.07); opacity:.86; }
        }
        @keyframes dsLetterD {
            0% { opacity:0; transform:scale(.62) rotateY(-24deg); filter:blur(3px); }
            1.5%, 5.5% { opacity:1; transform:scale(1) rotateY(0); filter:blur(0); }
            7.5%,100% { opacity:0; transform:scale(1.08); }
        }
        @keyframes dsLetterRest {
            0%, 2% { opacity:0; transform:translateX(-28px) scale(.92); filter:blur(3px); }
            4%, 6.5% { opacity:1; transform:translateX(0) scale(1); filter:blur(0); }
            8.5%,100% { opacity:0; transform:translateX(10px); }
        }
        @keyframes dsTagline {
            0%, 4.5% { opacity:0; transform:translate(-50%,7px); }
            5.5%, 7.5% { opacity:1; transform:translate(-50%,0); }
            9%,100% { opacity:0; }
        }
        @keyframes dsPortalLayer {
            0%, 6% { opacity:0; }
            7%, 12% { opacity:1; }
            14%,100% { opacity:0; }
        }
        @keyframes dsDoorOpen {
            0%, 6.5% { transform:rotateY(0) scale(.72); opacity:1; }
            7.5% { transform:rotateY(0) scale(1); opacity:1; }
            10.5% { transform:rotateY(-76deg) scale(2.05); opacity:1; }
            13%,100% { transform:rotateY(-94deg) scale(2.65); opacity:0; }
        }
        @keyframes dsProductReveal {
            0%, 10% { opacity:0; transform:scale(.88); filter:blur(3px); }
            13%, 98% { opacity:1; transform:scale(1); filter:blur(0); }
            100% { opacity:0; transform:scale(.98); }
        }
        @keyframes dsUploadScene {
            0%, 12% { opacity:0; }
            14%, 31% { opacity:1; transform:translateY(0); }
            33%,100% { opacity:0; }
        }
        @keyframes dsQuestionScene {
            0%, 31% { opacity:0; }
            34%, 50% { opacity:1; transform:translateY(0); }
            53%,100% { opacity:0; }
        }
        @keyframes dsChartScene {
            0%, 51% { opacity:0; }
            54%, 70% { opacity:1; transform:translateY(0); }
            73%,100% { opacity:0; }
        }
        @keyframes dsInsightsScene {
            0%, 71% { opacity:0; }
            74%, 98% { opacity:1; transform:translateY(0); }
            100% { opacity:0; }
        }
        @keyframes dsUploadProgress {
            0%, 14% { transform:scaleX(0); }
            28%,100% { transform:scaleX(1); }
        }
        @media (max-height:820px) and (min-width:901px){ .ds-portal-tour{height:520px}.ds-brand-d{font-size:126px}.ds-brand-rest span{font-size:62px}.ds-product-chart{height:175px} }
        @media (max-width:1100px){ .ds-portal-tour{height:520px}.ds-brand-stage{min-width:450px}.ds-brand-d{font-size:120px}.ds-brand-rest span{font-size:56px}.ds-product-sidebar{width:130px;flex-basis:130px}.ds-product-main{padding:22px 24px}.ds-product-scene{inset:22px 24px} }
</style>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.18, 0.82], gap="large")

    with left_col:
        _render_auth_showcase()

    with right_col:
        _, auth_center, _ = st.columns([0.14, 0.72, 0.14])
        with auth_center:
            _render_auth_assistant(mode)

            if mode == "welcome":
                if st.button("Create my account", type="primary", use_container_width=True, icon=":material/person_add:", key="auth_welcome_signup"):
                    _set_auth_mode("signup")
                    st.rerun()
                if st.button("I already have an account", use_container_width=True, icon=":material/login:", key="auth_welcome_login"):
                    _set_auth_mode("login")
                    st.rerun()
                st.markdown('<div class="ds-auth-note" style="text-align:center;">Forgot password?</div>', unsafe_allow_html=True)

            elif mode == "signup":
                st.markdown('<div class="ds-auth-label">CREATE ACCOUNT</div>', unsafe_allow_html=True)
                with st.form("datasense_sign_up_form"):
                    full_name = st.text_input("Full name", key="auth_sign_up_name", placeholder="Your name")
                    email = st.text_input("Email", key="auth_sign_up_email", placeholder="you@example.com")
                    password = st.text_input("Password", type="password", key="auth_sign_up_password")
                    confirm_password = st.text_input("Confirm password", type="password", key="auth_sign_up_confirm_password")
                    submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)

                if submitted:
                    full_name = full_name.strip()
                    email = email.strip()
                    if not full_name or not email or not password:
                        st.warning("Enter your name, email, and password.")
                    elif password != confirm_password:
                        st.warning("The passwords do not match.")
                    elif len(password) < 6:
                        st.warning("Use a password with at least 6 characters.")
                    else:
                        try:
                            response = sign_up_user(full_name, email, password)
                            user = getattr(response, "user", None)
                            session = getattr(response, "session", None)
                            if user is not None and session is not None:
                                _remember_authenticated_user(user)
                                st.session_state.auth_notice = "account_created"
                                st.rerun()
                            if user is not None:
                                st.session_state.auth_pending_email = email
                                st.session_state.auth_sign_in_email = email
                                _set_auth_mode("verify")
                                st.rerun()
                            st.error("DataSense could not create the account. Check the details and try again.")
                        except Exception as exc:
                            st.error(_friendly_auth_error(exc))

                if st.button("Sign in instead", use_container_width=True, key="auth_signup_login"):
                    _set_auth_mode("login")
                    st.rerun()
                if st.button("Back", use_container_width=True, key="auth_signup_back"):
                    _set_auth_mode("welcome")
                    st.rerun()

            elif mode == "verify":
                pending_email = str(st.session_state.get("auth_pending_email") or st.session_state.get("auth_sign_in_email") or "your email").strip()
                safe_email = html.escape(pending_email)
                st.markdown('<div class="ds-auth-label">CHECK YOUR EMAIL</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ds-auth-verify-banner">We sent a verification link to <strong>{safe_email}</strong>.<br><br>Open the newest DataSense or Supabase email, confirm your account, then come back here.</div>', unsafe_allow_html=True)
                st.markdown('<div class="ds-auth-steps-inline"><span>1. Open email</span><span>2. Confirm</span><span>3. Return</span></div>', unsafe_allow_html=True)
                if st.button("I’ve confirmed my email", type="primary", use_container_width=True, icon=":material/mark_email_read:", key="auth_verify_continue"):
                    _set_auth_mode("login")
                    st.rerun()
                if st.button("Back to start", use_container_width=True, key="auth_verify_different"):
                    for key in ("auth_pending_email", "auth_sign_in_email", "auth_sign_up_email", "auth_sign_up_name", "auth_sign_up_password", "auth_sign_up_confirm_password"):
                        st.session_state.pop(key, None)
                    _set_auth_mode("welcome")
                    st.rerun()

            else:
                pending_email = str(st.session_state.get("auth_pending_email") or st.session_state.get("auth_sign_in_email") or "").strip()
                if pending_email and "auth_sign_in_email" not in st.session_state:
                    st.session_state.auth_sign_in_email = pending_email

                st.markdown('<div class="ds-auth-label">SIGN IN</div>', unsafe_allow_html=True)
                with st.form("datasense_sign_in_form"):
                    email = st.text_input("Email", key="auth_sign_in_email", placeholder="you@example.com")
                    password = st.text_input("Password", type="password", key="auth_sign_in_password")
                    submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

                if submitted:
                    email = email.strip()
                    if not email or not password:
                        st.warning("Enter both your email and password.")
                    else:
                        try:
                            response = sign_in_user(email, password)
                            user = getattr(response, "user", None)
                            if user is None:
                                st.error("DataSense could not sign you in. Check your email and password.")
                            else:
                                _remember_authenticated_user(user)
                                st.session_state.pop("auth_pending_email", None)
                                st.rerun()
                        except Exception as exc:
                            st.error(_friendly_auth_error(exc))

                st.markdown('<div class="ds-auth-note">Forgot password?</div>', unsafe_allow_html=True)
                if st.button("Create account", use_container_width=True, key="auth_login_create"):
                    _set_auth_mode("signup")
                    st.rerun()
                if st.button("Back", use_container_width=True, key="auth_login_back"):
                    _set_auth_mode("welcome")
                    st.rerun()

def handle_sign_out() -> None:
    """Sign out, clear private state, then return to the welcome experience."""
    try:
        sign_out_user()
    finally:
        st.session_state.clear()
        st.session_state.auth_notice = "signed_out"
        st.session_state.auth_mode = "welcome"
    st.rerun()




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

if "dataset_source_mode" not in st.session_state:
    st.session_state.dataset_source_mode = "Upload CSV / Excel"

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

# Require authentication before rendering the DataSense workspace.
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if not st.session_state.auth_user:
    render_auth_screen()
    st.stop()

# Auth password fields are no longer rendered once the user is signed in. Clear
# any remaining widget values before the workspace starts.
for _auth_secret_key in (
    "auth_sign_in_password",
    "auth_sign_up_password",
    "auth_sign_up_confirm_password",
):
    st.session_state.pop(_auth_secret_key, None)


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

    # Keep account controls compact: the email and sign-out button are no
    # longer permanently visible. They live inside the collapsed account menu
    # rendered at the bottom of the sidebar.
    auth_user = st.session_state.get("auth_user") or {}

    if st.session_state.get("nav_active") == "Future Roadmap":
        st.session_state.nav_active = "Data Science Lab"
    elif "nav_active" not in st.session_state or st.session_state.nav_active in ("Home", "Chat"):
        st.session_state.nav_active = "Workspace"

    # Record one authenticated feature-open event when the user enters a
    # different DataSense section. Normal Streamlit reruns are de-duplicated.
    track_feature_open(
        feature=st.session_state.nav_active,
        auth_user=st.session_state.get("auth_user") or {},
    )

    nav_items = [
        ("Workspace", ":material/home:", None),
        ("Knowledge Base", ":material/library_books:", None),
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

    active_source = st.session_state.get("active_dataset_source")
    source_status = (
        f"{active_source} · Local AI"
        if active_source
        else "No dataset selected · Local AI"
    )

    st.markdown(
        f"""
        <div class="system-status">
            <span class="status-dot"></span>
            <div>
                <strong>Local AI online</strong>
                <small>{source_status}</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compact account control at the bottom of the sidebar. Profile is available
    # to every signed-in user; Admin Dashboard appears only for allow-listed admins.
    render_account_menu(
        auth_user,
        on_sign_out=handle_sign_out,
    )


# ==========================================================
# ACCOUNT ROUTER - Profile / Admin Dashboard
# ==========================================================

if st.session_state.nav_active == "Profile":
    render_profile_page(st.session_state.get("auth_user") or {})
    st.stop()

if st.session_state.nav_active == "Admin Dashboard":
    render_admin_dashboard(st.session_state.get("auth_user") or {})
    st.stop()


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
# KNOWLEDGE BASE - RAG
# ==========================================================

if st.session_state.nav_active == "Knowledge Base":
    retrieval_backend = configured_retrieval_backend()

    st.markdown("## Business Knowledge Base")
    st.caption(
        "Start with built-in KPI knowledge for an industry, or upload company-"
        "specific documentation. DataSense retrieves relevant passages before "
        "answering and keeps the retrieval index only for this session."
    )
    if retrieval_backend == "tfidf":
        st.caption("Retrieval mode: TF-IDF · Cloud-compatible text retrieval")
    else:
        st.caption(
            "Retrieval mode: Semantic embeddings · "
            f"Local Ollama model: {DEFAULT_EMBEDDING_MODEL}"
        )

    st.markdown("### Use built-in industry knowledge")
    try:
        starter_industries = available_starter_industries()
    except KnowledgeBaseError as exc:
        starter_industries = []
        st.error(str(exc))

    starter_industry = st.selectbox(
        "Industry",
        options=starter_industries,
        key="starter_knowledge_industry",
        help="Each starter pack contains five common KPI definitions and formulas.",
    ) if starter_industries else None

    st.markdown("### Or upload your own knowledge")
    knowledge_files = st.file_uploader(
        "Upload knowledge files",
        type=list(SUPPORTED_KNOWLEDGE_TYPES),
        accept_multiple_files=True,
        key="knowledge_uploader",
        help="Supported formats: PDF, TXT, Markdown, and CSV.",
    )

    with st.expander("Retrieval settings"):
        if retrieval_backend == "ollama":
            embedding_model = st.text_input(
                "Local Ollama embedding model",
                value=st.session_state.get(
                    "knowledge_embedding_model",
                    DEFAULT_EMBEDDING_MODEL,
                ),
                key="knowledge_embedding_model_input",
                help=(
                    "The default is embeddinggemma. Install it once with "
                    "`ollama pull embeddinggemma`."
                ),
            ).strip()
        else:
            embedding_model = DEFAULT_EMBEDDING_MODEL
            st.caption(
                "TF-IDF uses scikit-learn and does not require a separate "
                "embedding model or embedding API key."
            )

    starter_col, custom_col, clear_col = st.columns([2, 2, 1])
    with starter_col:
        starter_clicked = st.button(
            "Load starter knowledge",
            type="primary",
            use_container_width=True,
            disabled=starter_industry is None,
            icon=":material/database:",
        )
    with custom_col:
        custom_clicked = st.button(
            "Index uploaded files",
            use_container_width=True,
            disabled=not knowledge_files,
            icon=":material/upload_file:",
        )
    with clear_col:
        clear_clicked = st.button(
            "Clear",
            use_container_width=True,
            disabled="knowledge_index" not in st.session_state,
            icon=":material/delete:",
        )

    if clear_clicked:
        st.session_state.pop("knowledge_index", None)
        st.session_state.pop("knowledge_embedding_model", None)
        st.rerun()

    if starter_clicked or custom_clicked:
        if retrieval_backend == "ollama" and not embedding_model:
            st.error("Enter an Ollama embedding model name.")
        else:
            try:
                if starter_clicked:
                    documents = [starter_glossary_document(starter_industry)]
                    selected_starter = starter_industry
                else:
                    documents = [
                        (item.name, item.getvalue())
                        for item in knowledge_files
                    ]
                    selected_starter = None

                spinner_text = (
                    "Reading documents and creating semantic embeddings..."
                    if retrieval_backend == "ollama"
                    else "Reading documents and creating the TF-IDF index..."
                )
                with st.spinner(spinner_text):
                    index = build_knowledge_index(
                        documents,
                        model=embedding_model,
                        retrieval_backend=retrieval_backend,
                    )
                index["starter_industry"] = selected_starter
                st.session_state.knowledge_index = index
                if retrieval_backend == "ollama":
                    st.session_state.knowledge_embedding_model = embedding_model
                else:
                    st.session_state.pop("knowledge_embedding_model", None)
                st.success(
                    f"Knowledge ready · {len(index['documents'])} file(s) · "
                    f"{len(index['chunks'])} searchable chunks"
                )
            except KnowledgeBaseError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not build the knowledge index. Details: {exc}")

    knowledge_index = st.session_state.get("knowledge_index")
    if knowledge_index:
        st.markdown("### Indexed sources")
        if knowledge_index.get("starter_industry"):
            st.success(
                f"{knowledge_index['starter_industry']} starter knowledge is active."
            )
        for source_name in knowledge_index["documents"]:
            st.markdown(f"- `{source_name}`")
        index_backend = knowledge_index.get("backend", "ollama")
        if index_backend == "tfidf":
            retrieval_summary = "Retrieval: TF-IDF word and phrase matching"
        else:
            retrieval_summary = (
                f"Retrieval: Ollama semantic embeddings · "
                f"Model: {knowledge_index['model']}"
            )
        st.caption(
            f"{retrieval_summary} · "
            f"{len(knowledge_index['chunks'])} chunks · "
            "stored only for this session"
        )

        st.markdown("### Test retrieval")
        retrieval_question = st.text_input(
            "Question",
            placeholder="For example: How is allocation rate calculated?",
            key="knowledge_test_question",
        )
        if st.button(
            "Find relevant passages",
            disabled=not retrieval_question.strip(),
            key="test_knowledge_retrieval",
        ):
            try:
                matches = retrieve_knowledge(retrieval_question, knowledge_index)
                if not matches:
                    st.info("No sufficiently relevant passage was found.")
                for number, match in enumerate(matches, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"**Source {number}: {match['source']} · "
                            f"{match['location']}**"
                        )
                        st.caption(f"Relevance: {match['score']:.1%}")
                        st.write(match["text"])
            except KnowledgeBaseError as exc:
                st.error(str(exc))
    else:
        st.info(
            "Select an industry and click **Load starter knowledge**. No document "
            "upload is required. Use uploads only when you have organisation-"
            "specific definitions or policies."
        )

    st.stop()


# ==========================================================
# DATA SOURCE - FILE OR SUPABASE
# ==========================================================

if st.session_state.nav_active == "Data Science Lab":
    render_data_science_lab_intro()

st.markdown("### Choose a data source")
st.radio(
    "Data source",
    options=["Upload CSV / Excel", "Demo database"],
    horizontal=True,
    label_visibility="collapsed",
    key="dataset_source_mode",
    on_change=handle_dataset_source_change,
)

using_database = st.session_state.dataset_source_mode == "Demo database"

if using_database:
    source_col, refresh_col = st.columns([5, 1])
    with source_col:
        st.markdown("**Demo sales dataset**")
        st.caption(
            "DataSense uses Supabase when available and a bundled synthetic "
            "sample when database credentials are unavailable."
        )
    with refresh_col:
        st.button(
            "Refresh data",
            key="refresh_demo_database",
            use_container_width=True,
            icon=":material/refresh:",
        )

    try:
        database_df, source_info = load_demo_sales_with_fallback()
        dataset_id = dataframe_dataset_id(
            database_df,
            source_info["dataset_key"],
        )
        activate_dataset(database_df, dataset_id, source_info["label"])

        df = st.session_state.active_df.copy()
        uploaded_file_name = source_info["display_name"]
        uploaded_file_size = int(database_df.memory_usage(deep=True).sum())
        dataset_source_label = source_info["label"]
        st.session_state.database_last_refreshed_at = pd.Timestamp.now().strftime(
            "%d %b %Y, %I:%M:%S %p"
        )

        if source_info["mode"] == "database":
            st.success(f"Connected · {len(database_df):,} rows loaded")
            st.caption(
                "Last checked: "
                f"{st.session_state.database_last_refreshed_at} · "
                "Click Refresh data after changing Supabase."
            )
        else:
            st.warning(
                "Supabase is unavailable, so DataSense loaded its bundled "
                f"synthetic sample ({len(database_df):,} rows)."
            )
            st.caption(
                "This read-only sample contains no credentials or user data. "
                "Configure `.streamlit/secrets.toml` to use live Supabase."
            )
    except Exception as e:
        st.error(
            "Neither Supabase nor the bundled demo dataset could be loaded. "
            f"Details: {e}"
        )
        st.stop()

else:
    if st.session_state.nav_active == "Data Science Lab":
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
            </style>
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

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
        key="dataset_uploader",
        on_change=handle_dataset_uploader_change,
    )

    # The uploader widget is not the dataset's source of truth. Streamlit can
    # temporarily return None during navigation, so persist its bytes.
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
            render_data_science_lab(None, "no-dataset", show_intro=False)
        st.stop()

    dataset_id = f"file-{hashlib.sha256(file_bytes).hexdigest()[:16]}"

    try:
        if st.session_state.get("active_dataset_id") != dataset_id:
            if uploaded_file_name.lower().endswith(".csv"):
                loaded_df = pd.read_csv(BytesIO(file_bytes))
            else:
                loaded_df = pd.read_excel(BytesIO(file_bytes))

            activate_dataset(loaded_df, dataset_id, "Uploaded file")

        df = st.session_state.active_df.copy()
        dataset_source_label = "Uploaded file"

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
        <div class="dataset-card-badge">{dataset_source_label}</div>
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
            <div class="panel-row"><span class="panel-row-label">Source Name</span><span class="panel-row-value">{uploaded_file_name}</span></div>
            <div class="panel-row"><span class="panel-row-label">Source Type</span><span class="panel-row-value">{dataset_source_label}</span></div>
            <div class="panel-row"><span class="panel-row-label">Rows</span><span class="panel-row-value">{rows:,}</span></div>
            <div class="panel-row"><span class="panel-row-label">Columns</span><span class="panel-row-value">{columns}</span></div>
            <div class="panel-row"><span class="panel-row-label">Data Size</span><span class="panel-row-value">{file_size_mb:.2f} MB</span></div>
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
        for point in grounded_insights[:3]:
            safe_text = html.escape(point.get("summary") or point["text"])
            full_text = html.escape(point["text"], quote=True)
            insight_rows_html += (
                f'<div class="insight-row" title="{full_text}">{safe_text}</div>'
            )
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

def display_column_name(column) -> str:
    """Humanize machine-friendly dataset column names for result tables."""
    label = str(column).replace("_", " ").strip()
    return label.title() if label == label.lower() else label


def render_dataframe_message(message, fallback_key):
    """Render a saved calculation with a persistent Normal/3D chart choice."""
    result_frame = message["content"]
    plan = message.get("plan")

    if plan:
        analysis_title = str(plan.get("title", "Analysis"))
        if plan.get("analysis_type") == "top_bottom" and plan.get("measure"):
            measure_label = display_column_name(plan["measure"])
            if measure_label.lower() not in analysis_title.lower():
                analysis_title = f"{analysis_title} by {measure_label}"

        st.markdown(f"### {analysis_title}")

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

        if plan.get("analysis_type") == "patient_count" and len(result_frame) == 1:
            metric_column = result_frame.columns[0]
            metric_value = result_frame.iloc[0][metric_column]
            st.metric(display_column_name(metric_column), f"{int(metric_value):,}")

            audit = message.get("audit") or {}
            if audit.get("count_basis"):
                st.caption(f"Count basis: {audit['count_basis']}")

            render_calculation_audit(message)
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

        if plan.get("analysis_type") == "categorical_relationship":
            category = next(iter(plan.get("group_by") or []), "Category")
            measure = plan.get("measure") or "Value"
            average_column = f"Average {measure}"
            difference_column = "Difference from Overall %"
            eta_column = "Association (η²)"

            strength = str(result_frame["Association Strength"].iloc[0])
            eta_squared = float(result_frame[eta_column].iloc[0])
            explained_variation = eta_squared * 100
            explained_text = (
                "<0.01%"
                if explained_variation < 0.01
                else f"{explained_variation:.2f}%"
            )

            averages = pd.to_numeric(result_frame[average_column], errors="coerce")
            lowest_average = float(averages.min())
            highest_average = float(averages.max())
            average_gap = (
                (highest_average / lowest_average - 1) * 100
                if lowest_average != 0 else 0.0
            )

            if strength.lower() == "negligible":
                conclusion = (
                    f"No meaningful relationship was found. Average {measure} "
                    f"ranges from {lowest_average:,.2f} to {highest_average:,.2f} "
                    f"across {category} groups—a difference of only {average_gap:.2f}%."
                )
            else:
                conclusion = (
                    f"A {strength.lower()} relationship was found between "
                    f"{category} and {measure}. Group averages differ by "
                    f"{average_gap:.2f}%."
                )

            st.success(f"**Conclusion:** {conclusion}")
            st.caption(
                f"Because {category} is categorical, DataSense compared group "
                "distributions instead of using Pearson correlation."
            )

            association_col, variation_col, gap_col = st.columns(3)
            association_col.metric("Relationship", strength)
            variation_col.metric("Variation explained", explained_text)
            gap_col.metric("Largest average gap", f"{average_gap:.2f}%")

            compact_table = result_frame[
                [category, "Records", average_column, difference_column]
            ].copy()
            compact_table["Records"] = compact_table["Records"].map(
                lambda value: f"{int(value):,}"
            )
            compact_table[average_column] = compact_table[average_column].map(
                lambda value: f"{float(value):,.2f}"
            )
            compact_table[difference_column] = compact_table[difference_column].map(
                lambda value: f"{float(value):+.2f}%"
            )
            compact_table = compact_table.rename(
                columns={difference_column: "Vs overall"}
            )
            st.dataframe(compact_table, use_container_width=True, hide_index=True)

            with st.expander("Technical details"):
                st.write(
                    "Eta-squared (η²) measures how much numeric variation is "
                    "associated with differences between category-group averages. "
                    "Values below 0.01 are considered negligible."
                )
                st.code(f"η² = {eta_squared:.6f}")
                technical_columns = [
                    category,
                    f"Median {measure}",
                    f"Std Dev {measure}",
                ]
                st.dataframe(
                    result_frame[technical_columns],
                    use_container_width=True,
                    hide_index=True,
                )

            render_calculation_audit(message)
            return

    display_frame = result_frame.rename(columns=display_column_name)
    st.dataframe(display_frame, use_container_width=True, hide_index=True)
    render_calculation_audit(message)


def render_calculation_audit(message) -> None:
    """Expose enough evidence for a user to reproduce a calculated answer."""
    audit = message.get("audit")
    if not audit:
        return

    with st.expander("How this was calculated"):
        st.markdown("**Calculated by Pandas from the active dataset**")
        st.caption(
            "The local LLM planned the request; the displayed values were "
            "computed from dataset rows rather than generated as text."
        )

        row_col, used_col, group_col = st.columns(3)
        row_col.metric("Dataset rows", f"{audit['source_rows']:,}")
        used_col.metric("Valid rows used", f"{audit['valid_measure_rows']:,}")
        group_col.metric("Groups compared", f"{audit['groups_evaluated']:,}")

        st.markdown(f"**Formula:** `{audit['formula']}`")
        if audit.get("count_basis"):
            st.markdown(f"**Count basis:** `{audit['count_basis']}`")
        st.markdown(f"**Filters:** `{audit['filters']}`")
        st.caption(
            f"Rows after filters: {audit['rows_after_filters']:,} · "
            f"Sort: {audit['sort']} · Result limit: {audit['limit'] or 'All'}"
        )

        evidence = message.get("evidence")
        if isinstance(evidence, pd.DataFrame) and not evidence.empty:
            st.markdown("**Top comparison used to verify the winner**")
            st.dataframe(
                evidence.rename(columns=display_column_name),
                use_container_width=True,
                hide_index=True,
            )


def render_knowledge_sources(sources: list[dict] | None) -> None:
    """Show the exact retrieved passages used for a grounded answer."""
    if not sources:
        return

    with st.expander(f"Sources used ({len(sources)})"):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**{number}. {source['source']} · {source['location']}**"
            )
            st.caption(f"Retrieval relevance: {source['score']:.1%}")
            st.write(source["text"])


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
                render_knowledge_sources(message.get("sources"))


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

detected_intent = detect_intent(
    question,
    knowledge_available=bool(st.session_state.get("knowledge_index")),
)
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

elif intent == "KNOWLEDGE":

    task = question

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
                    "audit": build_calculation_audit(df, plan),
                    "message_id": f"{dataset_id}_{len(st.session_state.messages)}_{i}",
                }

                if (
                    plan.get("analysis_type") == "top_bottom"
                    and plan.get("limit") == 1
                ):
                    evidence_plan = dict(plan)
                    evidence_plan["limit"] = 5
                    evidence = calculate(df, evidence_plan)
                    if isinstance(evidence, pd.DataFrame):
                        analysis_message["evidence"] = evidence

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

                if (
                    plan.get("analysis_type") == "categorical_relationship"
                    and str(plan.get("chart") or "").lower() == "box"
                ):
                    category = next(iter(plan.get("group_by") or []), None)
                    measure = plan.get("measure")
                    if category in df.columns and measure in df.columns:
                        analysis_message["chart_data"] = df[[category, measure]].copy()

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

        knowledge_results = []
        knowledge_index = st.session_state.get("knowledge_index")

        if intent == "KNOWLEDGE" and not knowledge_index:
            response = (
                "I don't have a business knowledge base yet. Open **Knowledge "
                "Base**, upload a KPI glossary or supporting document, and build "
                "the local index first."
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })
            st.rerun()

        if knowledge_index and intent in {
            "KNOWLEDGE",
            "EXPLAIN",
            "KPI",
            "ANALYSIS",
        }:
            knowledge_results = retrieve_knowledge(
                question,
                knowledge_index,
                top_k=3,
            )

        if intent == "KNOWLEDGE" and not knowledge_results:
            response = (
                "I couldn't find enough support for that answer in the indexed "
                "knowledge. Add a relevant definition or document and rebuild "
                "the Knowledge Base."
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })
            st.rerun()

        retrieved_context = format_retrieved_context(knowledge_results)
        grounding_instructions = ""
        if retrieved_context:
            grounding_instructions = f"""

Retrieved Business Knowledge

{retrieved_context}

Grounding rules

- Use retrieved knowledge for business definitions, formulas, policies, and targets.
- Cite supporting passages inline as [Source 1], [Source 2], and so on.
- Do not claim that a source contains information that is not shown above.
- If the retrieved knowledge does not support part of the answer, say so clearly.
"""

        response = ask_llm(

            system_prompt,

            f"""
Dataset Information

{dataset_info}

{grounding_instructions}

Task

{task}
"""
        )

        st.session_state.messages.append({

            "role": "assistant",
            "content": response,
            "sources": knowledge_results,

        })

        with st.chat_message("assistant", avatar=chat_avatar("assistant")):

            st.write(response)
            render_knowledge_sources(knowledge_results)

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
