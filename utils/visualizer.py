import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def show_visuals(df):
    """
    Advanced Streamlit visualization module with upgraded charts.
    """
    st.title("📊 Advanced Data Visualization Dashboard")

    # --- Detect column types ---
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            continue

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # --- Sidebar Navigation ---
    with st.sidebar:
        st.header("🧭 Chart Navigator")
        chart_type = st.selectbox(
            "Choose Visualization Type",
            ["Histogram", "Heatmap", "Bar Chart", "Pie Chart", "Donut Chart",
             "Line Chart", "Scatter Plot", "Bubble Chart", "Pair Plot"]
        )
        st.markdown("---")

    st.subheader(f"📈 {chart_type}")

    # --- HISTOGRAM ---
    if chart_type == "Histogram":
        if not numeric_cols:
            st.warning("No numeric columns found.")
            return
        col = st.sidebar.selectbox("Select Numeric Column", numeric_cols)
        color = st.sidebar.selectbox("Group By (Color)", ["None"] + cat_cols)
        bins = st.sidebar.slider("Bins", 5, 150, 30)
        marginal = st.sidebar.selectbox("Marginal Plot", [None, 'box', 'violin', 'rug'])
        histnorm = st.sidebar.selectbox("Normalization", [None, 'percent', 'probability density'])
        
        fig = px.histogram(
            df, x=col, nbins=bins,
            color=color if color != "None" else None,
            marginal=marginal, histnorm=histnorm,
            title=f"Distribution of {col}", opacity=0.75
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[col].describe().to_frame("Statistics"))

    # --- HEATMAP ---
    elif chart_type == "Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns.")
            return
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

    # --- BAR CHART ---
    elif chart_type == "Bar Chart":
        if not cat_cols:
            st.warning("No categorical columns found.")
            return
        x = st.sidebar.selectbox("X-Axis (Category)", cat_cols)
        y = st.sidebar.selectbox("Y-Axis (Numeric)", numeric_cols)
        agg_func = st.sidebar.selectbox("Aggregation", ["Mean", "Sum", "Count"])
        if agg_func == "Mean":
            data = df.groupby(x)[y].mean().reset_index()
        elif agg_func == "Sum":
            data = df.groupby(x)[y].sum().reset_index()
        else:
            data = df[x].value_counts().reset_index()
            data.columns = [x, "Count"]
            y = "Count"
        fig = px.bar(data, x=x, y=y, title=f"{agg_func} of {y} by {x}", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- PIE / DONUT ---
    elif chart_type in ["Pie Chart", "Donut Chart"]:
        if not cat_cols:
            st.warning("No categorical columns found.")
            return
        col = st.sidebar.selectbox("Category Column", cat_cols)
        values = df[col].value_counts()
        top_n = st.sidebar.slider("Top Categories", 3, len(values), 10)
        values = values.nlargest(top_n)
        fig = px.pie(names=values.index, values=values.values, hole=0.4 if chart_type == "Donut Chart" else 0.0)
        st.plotly_chart(fig, use_container_width=True)

    # --- LINE CHART ---
    elif chart_type == "Line Chart":
        x = st.sidebar.selectbox("X-Axis", datetime_cols + numeric_cols + cat_cols)
        y = st.sidebar.selectbox("Y-Axis", numeric_cols)
        color = st.sidebar.selectbox("Color Group", ["None"] + cat_cols)
        fig = px.line(df, x=x, y=y, color=color if color != "None" else None, markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- SCATTER PLOT ---
    elif chart_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns.")
            return
        x = st.sidebar.selectbox("X-Axis", numeric_cols)
        y = st.sidebar.selectbox("Y-Axis", [c for c in numeric_cols if c != x])
        color = st.sidebar.selectbox("Color", ["None"] + cat_cols)
        fig = px.scatter(df, x=x, y=y, color=color if color != "None" else None)
        st.plotly_chart(fig, use_container_width=True)

    # --- BUBBLE CHART ---
    elif chart_type == "Bubble Chart":
        if len(numeric_cols) < 3:
            st.warning("Need at least 3 numeric columns.")
            return
        x = st.sidebar.selectbox("X-Axis", numeric_cols)
        y = st.sidebar.selectbox("Y-Axis", [c for c in numeric_cols if c != x])
        size = st.sidebar.selectbox("Bubble Size", [c for c in numeric_cols if c not in [x, y]])
        color = st.sidebar.selectbox("Color", ["None"] + cat_cols)
        fig = px.scatter(df, x=x, y=y, size=size, color=color if color != "None" else None, hover_name=color)
        st.plotly_chart(fig, use_container_width=True)

    # --- PAIR PLOT ---
    elif chart_type == "Pair Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns.")
            return
        selected_vars = st.sidebar.multiselect("Select Variables", numeric_cols, default=numeric_cols[:4])
        hue = st.sidebar.selectbox("Color By", ["None"] + cat_cols)
        if st.button("Generate Pair Plot"):
            fig = sns.pairplot(df[selected_vars + ([hue] if hue != "None" else [])], hue=hue if hue != "None" else None)
            st.pyplot(fig)
