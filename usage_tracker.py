"""Server-side feature usage tracking for DataSense AI.

R4 behavior:
- writes usage events with the Supabase secret key from Streamlit secrets;
- keeps normal user authentication on the publishable key in auth.py;
- records only user UUID, feature, action, and Supabase's created_at timestamp;
- de-duplicates normal Streamlit reruns while the user stays on one feature;
- fails fast for the rest of the browser session after one tracking error so
  analytics pages are not repeatedly slowed by a broken logging connection.

IMPORTANT:
The Supabase secret key must stay only in `.streamlit/secrets.toml` (and later
in Streamlit Cloud secrets). Never commit it to Git or expose it in browser
code.
"""

from __future__ import annotations

import streamlit as st
from supabase import create_client


USAGE_TABLE = "usage_events"
TRACKER_BUILD = "2026.08.21-USAGE-R4-SERVER-SIDE"


def _get_usage_admin_client():
    """Return a server-side Supabase client dedicated to usage logging."""
    client_key = "_usage_r4_admin_client"

    if client_key not in st.session_state:
        auth_secrets = st.secrets["supabase_auth"]
        url = str(auth_secrets["url"]).strip()
        secret_key = str(auth_secrets["secret_key"]).strip()

        if not url or not secret_key:
            raise RuntimeError(
                "Supabase usage tracking requires supabase_auth.url and "
                "supabase_auth.secret_key in Streamlit secrets."
            )

        st.session_state[client_key] = create_client(url, secret_key)

    return st.session_state[client_key]


def track_feature_open(
    feature: str,
    auth_user: dict,
    action: str = "opened",
) -> bool:
    """Record one feature-open event for the signed-in DataSense user.

    This function intentionally uses the server-side secret-key client rather
    than the user's publishable-key auth client. The authenticated user's UUID
    still comes from DataSense session state and is stored as ``user_id``.
    """
    normalized_feature = str(feature or "").strip()
    normalized_action = str(action or "opened").strip() or "opened"
    user_id = str((auth_user or {}).get("id") or "").strip()

    if not normalized_feature or not user_id:
        return False

    # If logging fails once, do not make every Streamlit rerun wait on another
    # failing network request. Signing out/restarting the browser session clears
    # this session-scoped flag.
    if st.session_state.get("_usage_r4_disabled_for_session"):
        return False

    event_key = f"{user_id}:{normalized_feature}:{normalized_action}"

    # Streamlit reruns frequently. Only log again after the user actually moves
    # to a different feature (and later comes back).
    if st.session_state.get("_usage_r4_last_event_key") == event_key:
        return False

    try:
        supabase = _get_usage_admin_client()
        supabase.table(USAGE_TABLE).insert(
            {
                "user_id": user_id,
                "feature": normalized_feature,
                "action": normalized_action,
            }
        ).execute()
    except Exception as exc:
        st.session_state["_usage_r4_error"] = str(exc)
        st.session_state["_usage_r4_disabled_for_session"] = True
        print(f"### USAGE TRACKING R4 DISABLED FOR SESSION: {exc} ###")
        return False

    st.session_state["_usage_r4_last_event_key"] = event_key
    st.session_state.pop("_usage_r4_error", None)
    print(
        "### USAGE TRACKED R4: "
        f"{normalized_feature} / {normalized_action} ###"
    )
    return True
