import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Expense Manager", page_icon="💸", layout="wide")
st.title("💸 Expense Manager")


with st.form("add_expense_form"):
    st.subheader("Add Expense")
    expense_date = st.date_input("Date", value=date.today())
    category = st.text_input("Category", placeholder="e.g. Food, Travel")
    amount = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Add Expense")

if submitted:
    payload = {
        "expense_date": expense_date.isoformat(),
        "category": category.strip(),
        "amount": amount,
    }
    if not payload["category"]:
        st.error("Category is required.")
    else:
        response = requests.post(f"{BACKEND_URL}/expenses", json=payload, timeout=10)
        if response.ok:
            st.success("Expense added successfully.")
        else:
            st.error(f"Failed to add expense: {response.text}")

st.divider()
st.subheader("Expense List")

try:
    response = requests.get(f"{BACKEND_URL}/expenses", timeout=10)
    response.raise_for_status()
    expenses = response.json()
except requests.RequestException as exc:
    st.error(f"Could not fetch expenses from backend: {exc}")
    expenses = []

if expenses:
    df = pd.DataFrame(expenses)
    df = df.rename(
        columns={
            "id": "ID",
            "expense_date": "Date",
            "category": "Category",
            "amount": "Amount",
        }
    )

    st.dataframe(df, use_container_width=True)

    chart_df = (
        df.groupby("Category", as_index=False)["Amount"]
        .sum()
        .sort_values("Amount", ascending=False)
    )
    st.subheader("Spending by Category")
    st.bar_chart(chart_df.set_index("Category")["Amount"])
else:
    st.info("No expenses found. Add your first expense above.")
