"""Profile and admin usage surfaces for DataSense AI.

This module keeps account-management code out of the large Streamlit app file.
It uses the Supabase secret key only on the Streamlit server for trusted profile
maintenance and admin reporting. The secret key must never be exposed in the UI
or committed to source control.
"""

from __future__ import annotations

from typing import Any
import html

import pandas as pd
import streamlit as st
from supabase import create_client


PROFILE_TABLE = "profiles"
USAGE_TABLE = "usage_events"
ACCOUNT_BUILD = "2026.08.21-ACCOUNT-R5"


def _get_server_client():
    """Return one server-side Supabase client for this Streamlit session."""
    client_key = "_account_r5_server_client"

    if client_key not in st.session_state:
        auth_secrets = st.secrets["supabase_auth"]
        url = str(auth_secrets["url"]).strip()
        secret_key = str(auth_secrets["secret_key"]).strip()

        if not url or not secret_key:
            raise RuntimeError(
                "Profile/Admin features require supabase_auth.url and "
                "supabase_auth.secret_key in Streamlit secrets."
            )

        st.session_state[client_key] = create_client(url, secret_key)

    return st.session_state[client_key]


def _normalise_admin_emails(value: Any) -> set[str]:
    """Convert Streamlit secret values into a lowercase email allow-list."""
    if value is None:
        return set()

    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        try:
            raw_items = list(value)
        except TypeError:
            raw_items = [value]

    return {
        str(item).strip().lower()
        for item in raw_items
        if str(item).strip()
    }


def configured_admin_emails() -> set[str]:
    """Read administrator emails from ``[admin].emails`` in Streamlit secrets."""
    try:
        admin_section = st.secrets["admin"]
        return _normalise_admin_emails(admin_section.get("emails", []))
    except Exception:
        return set()


def is_admin_user(auth_user: dict | None) -> bool:
    """Return True only when the signed-in email is explicitly allow-listed."""
    email = str((auth_user or {}).get("email") or "").strip().lower()
    return bool(email and email in configured_admin_emails())


def _profile_initials(name: str, email: str) -> str:
    source = (name or email or "User").strip()
    pieces = [piece for piece in source.replace("@", " ").split() if piece]
    if not pieces:
        return "U"
    if len(pieces) == 1:
        return pieces[0][:2].upper()
    return (pieces[0][:1] + pieces[1][:1]).upper()


def _format_created_at(value: Any) -> str:
    if value is None or value == "":
        return "Not available"

    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            return "Not available"
        return parsed.strftime("%d %b %Y")
    except Exception:
        return str(value)


def get_profile(user_id: str) -> dict:
    """Fetch one public profile row by authenticated user id."""
    user_id = str(user_id or "").strip()
    if not user_id:
        return {}

    response = (
        _get_server_client()
        .table(PROFILE_TABLE)
        .select("id,full_name,email,created_at")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    return dict(rows[0]) if rows else {}


def update_profile_name(user_id: str, full_name: str) -> None:
    """Update the public profile and matching Supabase Auth metadata."""
    user_id = str(user_id or "").strip()
    full_name = " ".join(str(full_name or "").split()).strip()

    if not user_id:
        raise ValueError("Missing authenticated user id.")
    if len(full_name) < 2:
        raise ValueError("Enter a name with at least 2 characters.")
    if len(full_name) > 100:
        raise ValueError("Name must be 100 characters or fewer.")

    client = _get_server_client()

    client.table(PROFILE_TABLE).update(
        {"full_name": full_name}
    ).eq("id", user_id).execute()

    # Keep Auth user_metadata in sync so the updated name is preserved after
    # sign-out/sign-in, where app.py rebuilds auth_user from user_metadata.
    client.auth.admin.update_user_by_id(
        user_id,
        {"user_metadata": {"full_name": full_name}},
    )


def render_account_menu(
    auth_user: dict,
    *,
    on_sign_out,
) -> None:
    """Render a compact, collapsed account control at the bottom of the sidebar."""
    full_name = str((auth_user or {}).get("full_name") or "").strip()
    email = str((auth_user or {}).get("email") or "").strip()
    display_name = full_name or (email.split("@", 1)[0] if email else "Account")

    st.markdown(
        '<div class="section-label" style="margin-top:22px;">Account</div>',
        unsafe_allow_html=True,
    )

    with st.expander(f"👤  {display_name}", expanded=False):
        if email:
            st.caption(email)

        if st.button(
            "Profile",
            key="account_open_profile",
            use_container_width=True,
            icon=":material/account_circle:",
        ):
            st.session_state.nav_active = "Profile"
            st.rerun()

        if is_admin_user(auth_user):
            if st.button(
                "Admin Dashboard",
                key="account_open_admin",
                use_container_width=True,
                icon=":material/admin_panel_settings:",
            ):
                st.session_state.nav_active = "Admin Dashboard"
                st.rerun()

        if st.button(
            "Sign out",
            key="account_sign_out",
            use_container_width=True,
            icon=":material/logout:",
        ):
            on_sign_out()


def render_profile_page(auth_user: dict) -> None:
    """Render the signed-in user's personal profile page."""
    user_id = str((auth_user or {}).get("id") or "").strip()
    session_email = str((auth_user or {}).get("email") or "").strip()
    session_name = str((auth_user or {}).get("full_name") or "").strip()

    st.markdown("## My Profile")
    st.caption("Manage the identity shown inside DataSense AI.")

    try:
        profile = get_profile(user_id)
    except Exception as exc:
        st.error(f"Could not load your profile. Details: {exc}")
        return

    full_name = str(profile.get("full_name") or session_name or "").strip()
    email = str(profile.get("email") or session_email or "").strip()
    created_at = _format_created_at(profile.get("created_at"))
    initials = html.escape(_profile_initials(full_name, email))
    safe_name = html.escape(full_name or "DataSense user")
    safe_email = html.escape(email)

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:16px;margin:0.6rem 0 1.1rem;">
            <div style="width:58px;height:58px;border-radius:50%;display:flex;
                        align-items:center;justify-content:center;font-weight:750;
                        font-size:1.05rem;background:rgba(105,88,255,.16);
                        border:1px solid rgba(128,115,255,.42);">
                {initials}
            </div>
            <div>
                <div style="font-size:1.15rem;font-weight:700;">{safe_name}</div>
                <div style="opacity:.72;font-size:.9rem;">{safe_email}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_left, info_right = st.columns(2)
    info_left.metric("Account", "Active")
    info_right.metric("Member since", created_at)

    st.markdown("### Personal details")
    with st.form("datasense_profile_form"):
        new_name = st.text_input(
            "Full name",
            value=full_name,
            max_chars=100,
        )
        st.text_input(
            "Email",
            value=email,
            disabled=True,
            help="Email changes use Supabase's confirmation flow and are not enabled here yet.",
        )
        save_clicked = st.form_submit_button(
            "Save profile",
            type="primary",
            use_container_width=True,
        )

    if save_clicked:
        try:
            update_profile_name(user_id, new_name)
            st.session_state.auth_user["full_name"] = " ".join(new_name.split()).strip()
            st.success("Profile updated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not update your profile. Details: {exc}")

    st.caption(
        "Your profile contains account identity information only. Uploaded datasets "
        "are not stored in your profile."
    )


def _fetch_admin_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch profiles and recent usage events for the admin dashboard."""
    client = _get_server_client()

    profiles_response = (
        client.table(PROFILE_TABLE)
        .select("id,full_name,email,created_at")
        .order("created_at", desc=True)
        .limit(1000)
        .execute()
    )
    usage_response = (
        client.table(USAGE_TABLE)
        .select("id,user_id,feature,action,created_at")
        .order("created_at", desc=True)
        .limit(1000)
        .execute()
    )

    profiles_df = pd.DataFrame(getattr(profiles_response, "data", None) or [])
    usage_df = pd.DataFrame(getattr(usage_response, "data", None) or [])
    return profiles_df, usage_df


def _prepare_recent_activity(
    profiles_df: pd.DataFrame,
    usage_df: pd.DataFrame,
) -> pd.DataFrame:
    if usage_df.empty:
        return pd.DataFrame(columns=["Name", "Email", "Feature", "Action", "Time"])

    activity = usage_df.copy()

    if not profiles_df.empty and "id" in profiles_df.columns:
        profile_columns = [
            column
            for column in ["id", "full_name", "email"]
            if column in profiles_df.columns
        ]
        activity = activity.merge(
            profiles_df[profile_columns],
            left_on="user_id",
            right_on="id",
            how="left",
            suffixes=("", "_profile"),
        )

    activity["Name"] = activity.get("full_name", pd.Series(index=activity.index, dtype="object")).fillna("Unknown user")
    activity["Email"] = activity.get("email", pd.Series(index=activity.index, dtype="object")).fillna("")
    activity["Feature"] = activity.get("feature", pd.Series(index=activity.index, dtype="object")).fillna("")
    activity["Action"] = activity.get("action", pd.Series(index=activity.index, dtype="object")).fillna("")
    timestamps = pd.to_datetime(activity.get("created_at"), errors="coerce", utc=True)
    activity["Time"] = timestamps.dt.strftime("%d %b %Y %H:%M UTC").fillna("")

    return activity[["Name", "Email", "Feature", "Action", "Time"]]


def _prepare_user_summary(
    profiles_df: pd.DataFrame,
    usage_df: pd.DataFrame,
) -> pd.DataFrame:
    if profiles_df.empty:
        return pd.DataFrame(columns=["Name", "Email", "Feature opens", "Last active", "Joined"])

    profiles = profiles_df.copy()
    profiles["Name"] = profiles.get("full_name", pd.Series(index=profiles.index, dtype="object")).fillna("")
    profiles["Email"] = profiles.get("email", pd.Series(index=profiles.index, dtype="object")).fillna("")
    joined = pd.to_datetime(profiles.get("created_at"), errors="coerce", utc=True)
    profiles["Joined"] = joined.dt.strftime("%d %b %Y").fillna("")

    if usage_df.empty:
        profiles["Feature opens"] = 0
        profiles["Last active"] = "Never"
        return profiles[["Name", "Email", "Feature opens", "Last active", "Joined"]]

    usage = usage_df.copy()
    usage["created_at"] = pd.to_datetime(usage.get("created_at"), errors="coerce", utc=True)
    summary = (
        usage.groupby("user_id", dropna=False)
        .agg(
            **{
                "Feature opens": ("id", "count"),
                "Last active raw": ("created_at", "max"),
            }
        )
        .reset_index()
    )
    summary["Last active"] = summary["Last active raw"].dt.strftime("%d %b %Y %H:%M UTC")

    profiles = profiles.merge(summary, left_on="id", right_on="user_id", how="left")
    profiles["Feature opens"] = profiles["Feature opens"].fillna(0).astype(int)
    profiles["Last active"] = profiles["Last active"].fillna("Never")

    return profiles[["Name", "Email", "Feature opens", "Last active", "Joined"]]


def render_admin_dashboard(auth_user: dict) -> None:
    """Render a read-only product-usage dashboard for allow-listed admins."""
    if not is_admin_user(auth_user):
        st.error("You do not have permission to view the Admin Dashboard.")
        return

    st.markdown("## Admin Dashboard")
    st.caption(
        "Registered users and DataSense feature usage. The dashboard is read-only "
        "and uses the server-side Supabase secret key."
    )

    refresh_col, note_col = st.columns([1, 4])
    with refresh_col:
        refresh_clicked = st.button(
            "Refresh",
            icon=":material/refresh:",
            use_container_width=True,
            key="admin_dashboard_refresh",
        )
    with note_col:
        st.caption("Shows up to the latest 1,000 usage events.")

    if refresh_clicked:
        st.session_state.pop("_admin_dashboard_data", None)

    try:
        if "_admin_dashboard_data" not in st.session_state:
            st.session_state._admin_dashboard_data = _fetch_admin_data()
        profiles_df, usage_df = st.session_state._admin_dashboard_data
    except Exception as exc:
        st.error(f"Could not load admin analytics. Details: {exc}")
        return

    total_users = int(len(profiles_df))
    total_events = int(len(usage_df))
    active_users = int(usage_df["user_id"].nunique()) if not usage_df.empty and "user_id" in usage_df else 0

    most_used_feature = "No activity yet"
    if not usage_df.empty and "feature" in usage_df:
        feature_counts = usage_df["feature"].dropna().astype(str).value_counts()
        if not feature_counts.empty:
            most_used_feature = str(feature_counts.index[0])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Registered users", total_users)
    metric_cols[1].metric("Active users", active_users)
    metric_cols[2].metric("Feature opens", total_events)
    metric_cols[3].metric("Most used feature", most_used_feature)

    overview_tab, users_tab, activity_tab = st.tabs(
        ["Overview", "Users", "Recent activity"]
    )

    with overview_tab:
        st.markdown("### Feature usage")
        if usage_df.empty or "feature" not in usage_df:
            st.info("No usage events have been recorded yet.")
        else:
            feature_usage = (
                usage_df["feature"]
                .dropna()
                .astype(str)
                .value_counts()
                .rename_axis("Feature")
                .reset_index(name="Opens")
            )
            st.bar_chart(feature_usage.set_index("Feature"))
            st.dataframe(feature_usage, use_container_width=True, hide_index=True)

    with users_tab:
        st.markdown("### Registered users")
        user_summary = _prepare_user_summary(profiles_df, usage_df)
        if user_summary.empty:
            st.info("No registered profiles found.")
        else:
            st.dataframe(user_summary, use_container_width=True, hide_index=True)

    with activity_tab:
        st.markdown("### Latest feature activity")
        recent_activity = _prepare_recent_activity(profiles_df, usage_df)
        if recent_activity.empty:
            st.info("No usage events have been recorded yet.")
        else:
            st.dataframe(
                recent_activity.head(250),
                use_container_width=True,
                hide_index=True,
            )
