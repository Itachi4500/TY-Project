import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

def show_hypothesis_testing(df):
    """
    Smart Hypothesis Testing Suite with Normality/Variance visual checks 
    and Auto Test Recommendations.
    """
    st.title("🧮 Smart Hypothesis Testing Suite")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    test_type = st.sidebar.selectbox("Select Test", [
        "Independent T-test", "Paired T-test", "Welch’s T-test", 
        "Mann–Whitney U Test", "ANOVA (One-Way)", "Chi-square Test"
    ])

    # Helper
    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1)+(ny-1)*np.var(y, ddof=1))/(nx+ny-2))
        return (np.mean(x)-np.mean(y))/pooled_std

    def plot_qq(data, title):
        fig = plt.figure()
        sm.qqplot(data, line='45', fit=True)
        plt.title(title)
        st.pyplot(fig)

    def variance_plot(group1, group2, label1, label2):
        fig, ax = plt.subplots()
        ax.boxplot([group1, group2], labels=[label1, label2])
        plt.title("Variance Comparison")
        st.pyplot(fig)

    # --- Independent T-test ---
    if test_type == "Independent T-test":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Grouping Variable (2 groups)", cat_cols)
        groups = df[cat].dropna().unique()
        if len(groups) != 2:
            st.warning("Categorical variable must have exactly 2 unique values.")
            return

        g1 = df[df[cat] == groups[0]][num].dropna()
        g2 = df[df[cat] == groups[1]][num].dropna()

        st.pyplot(sns.boxplot(x=df[cat], y=df[num]))

        st.markdown("### 🔍 Assumption Checks")
        st.write("**Normality (Shapiro-Wilk Test):**")
        stat1, p1 = stats.shapiro(g1)
        stat2, p2 = stats.shapiro(g2)
        st.write(f"{groups[0]} → p = {p1:.4f}, {groups[1]} → p = {p2:.4f}")
        st.write("✅ Normal if p ≥ 0.05")

        plot_qq(g1, f"QQ Plot - {groups[0]}")
        plot_qq(g2, f"QQ Plot - {groups[1]}")

        st.write("**Variance Equality (Levene’s Test):**")
        lev_stat, lev_p = stats.levene(g1, g2)
        st.write(f"Levene’s p = {lev_p:.4f}")
        variance_plot(g1, g2, groups[0], groups[1])

        # 🧠 Auto Recommendation
        st.markdown("### 🧠 Recommended Test:")
        if p1 < 0.05 or p2 < 0.05:
            st.info("Data is not normal → Use **Mann–Whitney U Test** (Non-parametric)")
        elif lev_p < 0.05:
            st.info("Unequal variances → Use **Welch’s T-test**")
        else:
            st.success("Normal data + equal variances → Use **Independent T-test**")

        if st.button("Run T-test"):
            t, p = stats.ttest_ind(g1, g2)
            st.success(f"T = {t:.4f}, P = {p:.4f}, Cohen’s d = {cohen_d(g1, g2):.3f}")

    # --- Paired T-test ---
    elif test_type == "Paired T-test":
        x = st.sidebar.selectbox("Variable 1", numeric_cols)
        y = st.sidebar.selectbox("Variable 2 (paired)", [c for c in numeric_cols if c != x])
        diff = df[x] - df[y]
        st.write("### 🔍 Normality Check for Differences")
        stat, p = stats.shapiro(diff.dropna())
        plot_qq(diff.dropna(), "QQ Plot of Differences")
        st.write(f"p = {p:.4f}")
        if p < 0.05:
            st.info("Data not normal → Consider **Wilcoxon Signed-Rank Test**.")
        if st.button("Run Paired T-test"):
            t, p = stats.ttest_rel(df[x], df[y])
            st.success(f"T = {t:.4f}, P = {p:.4f}")

    # --- ANOVA ---
    elif test_type == "ANOVA (One-Way)":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Categorical Variable (3+ groups)", cat_cols)
        groups = [g[num].dropna() for n, g in df.groupby(cat)]
        if len(groups) < 3:
            st.warning("Need at least 3 groups for ANOVA.")
            return
        if st.button("Run ANOVA"):
            f, p = stats.f_oneway(*groups)
            st.success(f"F = {f:.4f}, P = {p:.4f}")
            sns.boxplot(x=df[cat], y=df[num])
            st.pyplot(plt)
            if p < 0.05:
                st.info("Significant differences found. Consider post-hoc Tukey test.")

    # --- Chi-square ---
    elif test_type == "Chi-square Test":
        cat1 = st.sidebar.selectbox("Categorical Variable 1", cat_cols)
        cat2 = st.sidebar.selectbox("Categorical Variable 2", [c for c in cat_cols if c != cat1])
        if st.button("Run Chi-square Test"):
            table = pd.crosstab(df[cat1], df[cat2])
            chi2, p, dof, ex = stats.chi2_contingency(table)
            st.success(f"Chi² = {chi2:.3f}, P = {p:.4f}, DF = {dof}")
            st.dataframe(pd.DataFrame(ex, index=table.index, columns=table.columns))
