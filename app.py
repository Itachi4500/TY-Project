import streamlit as st
import warnings
import pandas as pd
import numpy as np

from utils.upload import upload_data
from utils.cleaner import clean_data
from utils.eda import run_eda
from utils.visualizer import show_visuals
from utils.modeler import run_modeling
from utils.exporter import export_data
from utils.memory import remember, recall, forget, clear_all_memory, show_memory, show_memory_history
from utils.powerbi_pipeline import powerbi_pipeline
from utils.refresh import refresh_data

warnings.filterwarnings("ignore")
st.set_page_config(page_title="🧠 Enhanced Data Analysis Assistant", layout="wide")
st.title("🧠 Enhanced Data Analysis Assistant")

# --- Session Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None

# --- Advanced Sidebar Navigation ---
nav = st.sidebar.radio("📌 Navigation", [
    "Refresh",
    "Upload Data",
    "Data Cleaning",
    "EDA",
    "Visualizations",
    "Model Training",
    "Power BI Pipeline",
    "Memory & Notes",
    "Export"
])

# --- Navigation Routing ---
if nav == "Upload Data":
    df = upload_data()
    if df is not None:
        st.session_state.df = df
        st.dataframe(df.head())

elif nav == "Refresh":
    refresh_data()

elif nav == "Data Cleaning":
    if st.session_state.df is not None:
        df_cleaned = clean_data(st.session_state.df)
        st.session_state.df = df_cleaned
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "EDA":
    if st.session_state.df is not None:
        run_eda(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Visualizations":
    if st.session_state.df is not None:
        show_visuals(st.session_state.df)
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
    show_memory()
    show_memory_history()
    with st.expander("➕ Add Note"):
        key = st.text_input("Memory Key (e.g. 'notes.data.cleaning')")
        value = st.text_area("Memory Value")
        if st.button("💾 Remember"):
            remember(key, value)
    with st.expander("❌ Forget Note"):
        forget_key = st.text_input("Key to forget")
        if st.button("Forget"):
            forget(forget_key)
    if st.button("🧹 Clear All Memory"):
        clear_all_memory()

elif nav == "Export":
    if st.session_state.df is not None:
        export_data(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")
