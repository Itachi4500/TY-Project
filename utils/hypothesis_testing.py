import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

def show_hypothesis_testing(df):
    """
    A fully modular hypothesis testing dashboard.
    """
    st.title("🧮 Hypothesis Testing Suite")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    test_type = st.sidebar.selectbox("Choose Test Type", [
        "Independent T-test", "Paired T-test", "Welch’s T-test", 
        "Mann–Whitney U Test", "ANOVA (One-Way)", "Chi-square Test"
    ])

    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1)+(ny-1)*np.var(y, ddof=1))/(nx+ny-2))
        return (np.mean(x)-np.mean(y))/pooled_std

    # --- Independent T-test ---
    if test_type == "Independent T-test":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Categorical Variable", cat_cols)
        if num and cat and df[cat].nunique() == 2:
            groups = df[cat].dropna().unique()
            g1 = df[df[cat] == groups[0]][num].dropna()
            g2 = df[df[cat] == groups[1]][num].dropna()
            st.pyplot(sns.boxplot(x=df[cat], y=df[num]))
            if st.button("Run T-test"):
                t, p = stats.ttest_ind(g1, g2)
                st.success(f"T = {t:.4f}, P = {p:.4f}, Cohen's d = {cohen_d(g1,g2):.3f}")

    # --- Paired T-test ---
    elif test_type == "Paired T-test":
        x = st.sidebar.selectbox("Variable 1", numeric_cols)
        y = st.sidebar.selectbox("Variable 2", [c for c in numeric_cols if c != x])
        if st.button("Run Paired T-test"):
            t, p = stats.ttest_rel(df[x], df[y])
            st.success(f"T = {t:.4f}, P = {p:.4f}")
            st.write(f"Mean Diff = {(df[x]-df[y]).mean():.4f}")

    # --- ANOVA ---
    elif test_type == "ANOVA (One-Way)":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Categorical Variable", cat_cols)
        if st.button("Run ANOVA"):
            samples = [group[num].dropna() for name, group in df.groupby(cat)]
            f, p = stats.f_oneway(*samples)
            st.success(f"F = {f:.4f}, P = {p:.4f}")
            sns.boxplot(x=df[cat], y=df[num])
            st.pyplot(plt)

    # --- Chi-square Test ---
    elif test_type == "Chi-square Test":
        cat1 = st.sidebar.selectbox("Category 1", cat_cols)
        cat2 = st.sidebar.selectbox("Category 2", [c for c in cat_cols if c != cat1])
        if st.button("Run Chi-square Test"):
            table = pd.crosstab(df[cat1], df[cat2])
            chi2, p, dof, ex = stats.chi2_contingency(table)
            st.success(f"Chi² = {chi2:.3f}, P = {p:.4f}, DF = {dof}")
            st.dataframe(pd.DataFrame(ex, index=table.index, columns=table.columns))
