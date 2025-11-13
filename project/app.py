import streamlit as st
import warnings
import pandas as pd
import numpy as np

# --- Local Imports ---
# from api_server import app
from utils.upload import upload_data
from utils.cleaner import clean_data
from utils.eda import run_eda
from utils.visualizer import show_visuals  
from utils.hypothesis_testing import show_hypothesis_testing  
from utils.modeler import run_modeling
from utils.exporter import export_data
from utils.memory import (
    remember, recall, forget, clear_all_memory,
    show_memory, show_memory_history
)
from utils.powerbi_pipeline import powerbi_pipeline
from utils.refresh import refresh_data

# Optional: Authentication (can be re-enabled later)
# from utils.login import login_page
# from utils.auth import check_auth

# --- Configuration ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Smart Data Analysis Assistant: Simplified with AI", layout="wide")
st.title("Smart Data Analysis Assistant: Simplified with AI")

# --- Session Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None

# Optional: Authentication
# if not check_auth():
#     login_page()
#     st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("📂 Main Navigation")
nav = st.sidebar.radio(
    "Go to Section:",
    [
        "Refresh",
        "Upload Data",
        "Data Cleaning",
        "EDA (Exploratory Analysis)",
        "Visualizations",
        "Hypothesis Testing",
        "Model Training",
        "Power BI Pipeline",
        "Memory & Notes",
        "Export"
    ],
)

# --- Navigation Routing ---
if nav == "Refresh":
    refresh_data()

elif nav == "Upload Data":
    df = upload_data()
    if df is not None:
        st.session_state.df = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

elif nav == "Data Cleaning":
    if st.session_state.df is not None:
        st.session_state.df = clean_data(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "EDA (Exploratory Analysis)":
    if st.session_state.df is not None:
        run_eda(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Visualizations":
    if st.session_state.df is not None:
        show_visuals(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Hypothesis Testing":
    if st.session_state.df is not None:
        show_hypothesis_testing(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Model Training":
    if st.session_state.df is not None:
        run_modeling(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Power BI Pipeline":
    if st.session_state.df is not None:
        powerbi_pipeline(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Memory & Notes":
    st.subheader("🧠 Memory & Notes Center")
    st.markdown("Use this section to store or recall analysis notes, key results, or reminders.")
    
    # --- Memory Overview ---
    show_memory()
    show_memory_history()

    # --- Add Note ---
    with st.expander("➕ Add Note to Memory"):
        key = st.text_input("🗝️ Memory Key (e.g. 'notes.data.cleaning')")
        value = st.text_area("🧾 Memory Value")
        if st.button("💾 Save Note"):
            if key and value:
                remember(key, value)
                st.success(f"Saved memory under key: `{key}`")
            else:
                st.warning("Please provide both key and value.")

    # --- Forget Note ---
    with st.expander("❌ Forget Specific Note"):
        forget_key = st.text_input("Key to forget")
        if st.button("🗑 Forget Note"):
            if forget_key:
                forget(forget_key)
                st.info(f"Forgot memory with key: `{forget_key}`")

    # --- Clear All Memory ---
    if st.button("🧹 Clear All Memory"):
        clear_all_memory()
        st.warning("All memory cleared.")

elif nav == "Export":
    if st.session_state.df is not None:
        export_data(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:golden;'>"
    "🧩 Made by Prem And Manoj"
    "</p>",
    unsafe_allow_html=True,
)
