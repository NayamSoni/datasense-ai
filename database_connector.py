import pandas as pd
import streamlit as st


def test_database_connection() -> bool:
    """Check whether DataSense AI can connect to Supabase."""
    conn = st.connection("datasense_db", type="sql")

    result = conn.query(
        "SELECT 1 AS connection_ok",
        ttl=0,
    )

    return result.iloc[0]["connection_ok"] == 1


def load_demo_sales() -> pd.DataFrame:
    """Load the latest demo_sales data from Supabase."""
    conn = st.connection("datasense_db", type="sql")

    df = conn.query(
        """
        SELECT *
        FROM public.demo_sales
        ORDER BY order_date, order_id
        """,
        ttl=0,
    )

    return df