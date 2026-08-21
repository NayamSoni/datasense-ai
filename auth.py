import streamlit as st
from supabase import Client, create_client


def get_supabase_client() -> Client:
    """Return a Supabase client for the current Streamlit user session."""

    if "supabase_client" not in st.session_state:
        url = st.secrets["supabase_auth"]["url"]
        key = st.secrets["supabase_auth"]["publishable_key"]

        st.session_state.supabase_client = create_client(url, key)

    return st.session_state.supabase_client

def sign_up_user(full_name: str, email: str, password: str):
    """Create a new DataSense user."""

    supabase = get_supabase_client()

    return supabase.auth.sign_up(
        {
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name
                }
            },
        }
    )


def sign_in_user(email: str, password: str):
    """Sign in an existing DataSense user."""

    supabase = get_supabase_client()

    response = supabase.auth.sign_in_with_password(
        {
            "email": email,
            "password": password,
        }
    )

    return response


def sign_out_user():
    """Sign out the current DataSense user."""

    supabase = get_supabase_client()
    supabase.auth.sign_out()