import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import time

def live_dashboard(df):
    st.set_page_config(layout="wide")  # Ensures full width is used
    st.title("📡 Live Dashboard")

    # ----------------------- Page Size Settings ------------------------
    with st.expander("🧩 Dashboard Page Size Configuration"):
        size_option = st.selectbox("Choose Page Size", ["Default (1280x720)", "4:3 (960x720)", "Letter (816x1056)", "Custom Size"])
        
        if size_option == "Default (1280x720)":
            width, height = 1280, 720
        elif size_option == "4:3 (960x720)":
            width, height = 960, 720
        elif size_option == "Letter (816x1056)":
            width, height = 816, 1056
        else:
            width = st.number_input("Custom Width", min_value=400, max_value=3840, value=1280, step=10)
            height = st.number_input("Custom Height", min_value=300, max_value=2160, value=720, step=10)

        st.markdown(f"**Selected Size:** `{width} x {height}`")

    # ------------------------- KPI Section ----------------------------
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # -------------------- Live Simulation Option ----------------------
    with st.expander("🔁 Simulate Live Data Refresh"):
        interval = st.slider("Refresh Interval (seconds)", 1, 10, 5)
        simulate = st.checkbox("Start Simulation")
        if simulate:
            for i in range(5):
                st.info(f"⏳ Refreshing data... {i+1}")
                time.sleep(interval)
                st.rerun()

    st.divider()

    # --------------------------- Filters ------------------------------
    st.subheader("🎯 Filter Data")
    filter_column = st.selectbox("Select Column to Filter", ["None"] + list(df.columns))
    if filter_column != "None":
        unique_vals = df[filter_column].dropna().unique()
        selected = st.multiselect(f"Filter values in {filter_column}", unique_vals, default=unique_vals)
        df = df[df[filter_column].isin(selected)]

    st.divider()

    # -------------------------- Chart Builder --------------------------
    st.subheader("📈 Visual Insights")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        x_col = st.selectbox("X-axis Column", df.columns)
        y_col = st.selectbox("Y-axis Column", df.select_dtypes(include=np.number).columns)

    with chart_col2:
        chart_type = st.radio("Chart Type", ["Bar", "Line", "Scatter", "Pie"])

    fig = None
    if chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", width=width, height=height)
    elif chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", width=width, height=height)
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}", width=width, height=height)
    elif chart_type == "Pie":
        fig = px.pie(df, names=x_col, title=f"Pie Chart of {x_col}", width=width, height=height)

    if fig:
        st.plotly_chart(fig, use_container_width=False)

    # -------------------------- Top N Section --------------------------
    st.subheader("📌 Top N Categories")
    group_col = st.selectbox("Group by column", df.columns)
    metric_col = st.selectbox("Metric column", df.select_dtypes(include=np.number).columns)
    top_n = st.slider("Top N", 1, 20, 5)

    top_df = df.groupby(group_col)[metric_col].sum().sort_values(ascending=False).head(top_n).reset_index()
    st.dataframe(top_df)

    st.success("✅ Live Dashboard Ready for Post-Analysis Monitoring!")
