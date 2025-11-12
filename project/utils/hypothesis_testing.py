import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def show_hypothesis_testing(df):
    """
    Smart Hypothesis Testing Suite with Normality/Variance visual checks,
    Auto Test Recommendations, and Z-tests (mean, proportion) + Two-Way ANOVA.
    """
    st.title("🧮 Smart Hypothesis Testing Suite")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    test_type = st.sidebar.selectbox("Select Statistical Test", [
        "Independent T-test", "Paired T-test", "Welch’s T-test", 
        "Mann–Whitney U Test", "ANOVA (One-Way)", "ANOVA (Two-Way)",
        "Z-test for Mean", "Z-test for Proportion", "Chi-square Test"
    ])

    # --- Helper Functions ---
    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1)+(ny-1)*np.var(y, ddof=1))/(nx+ny-2))
        return (np.mean(x)-np.mean(y))/pooled_std

    def interpretation(p_value, alpha=0.05):
        """Returns hypothesis interpretation message"""
        if p_value < alpha:
            st.error(f"❌ Null Hypothesis Rejected (p = {p_value:.4f} < 0.05)")
            st.markdown("**Interpretation:** There is a *statistically significant difference/effect.*")
        else:
            st.success(f"✅ Fail to Reject Null Hypothesis (p = {p_value:.4f} ≥ 0.05)")
            st.markdown("**Interpretation:** No *statistically significant difference/effect* detected.")

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
        stat1, p1 = stats.shapiro(g1)
        stat2, p2 = stats.shapiro(g2)
        st.write(f"{groups[0]} → p = {p1:.4f}, {groups[1]} → p = {p2:.4f}")
        plot_qq(g1, f"QQ Plot - {groups[0]}")
        plot_qq(g2, f"QQ Plot - {groups[1]}")

        lev_stat, lev_p = stats.levene(g1, g2)
        st.write(f"Levene’s p = {lev_p:.4f}")
        variance_plot(g1, g2, groups[0], groups[1])

        if st.button("Run Independent T-test"):
            t, p = stats.ttest_ind(g1, g2)
            st.info(f"T = {t:.4f}, P = {p:.4f}, Cohen’s d = {cohen_d(g1, g2):.3f}")
            interpretation(p)

    # --- Paired T-test ---
    elif test_type == "Paired T-test":
        x = st.sidebar.selectbox("Variable 1", numeric_cols)
        y = st.sidebar.selectbox("Variable 2 (paired)", [c for c in numeric_cols if c != x])
        diff = df[x] - df[y]
        stat, p = stats.shapiro(diff.dropna())
        plot_qq(diff.dropna(), "QQ Plot of Differences")
        st.write(f"Shapiro p = {p:.4f}")
        if st.button("Run Paired T-test"):
            t, p = stats.ttest_rel(df[x], df[y])
            st.info(f"T = {t:.4f}, P = {p:.4f}")
            interpretation(p)

    # --- Welch’s T-test ---
    elif test_type == "Welch’s T-test":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Grouping Variable (2 groups)", cat_cols)
        groups = df[cat].dropna().unique()
        if len(groups) != 2:
            st.warning("Must have exactly 2 groups.")
            return
        g1 = df[df[cat] == groups[0]][num].dropna()
        g2 = df[df[cat] == groups[1]][num].dropna()
        if st.button("Run Welch’s T-test"):
            t, p = stats.ttest_ind(g1, g2, equal_var=False)
            st.info(f"T = {t:.4f}, P = {p:.4f}")
            interpretation(p)

    # --- Mann–Whitney U Test ---
    elif test_type == "Mann–Whitney U Test":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Grouping Variable (2 groups)", cat_cols)
        groups = df[cat].dropna().unique()
        if len(groups) != 2:
            st.warning("Must have exactly 2 groups.")
            return
        g1 = df[df[cat] == groups[0]][num].dropna()
        g2 = df[df[cat] == groups[1]][num].dropna()
        if st.button("Run Mann–Whitney Test"):
            u, p = stats.mannwhitneyu(g1, g2)
            st.info(f"U = {u:.4f}, P = {p:.4f}")
            interpretation(p)

    # --- One-Way ANOVA ---
    elif test_type == "ANOVA (One-Way)":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Categorical Variable (3+ groups)", cat_cols)
        if st.button("Run ANOVA (One-Way)"):
            model = ols(f"{num} ~ C({cat})", data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.dataframe(anova_table)
            f = anova_table["F"][0]
            p = anova_table["PR(>F)"][0]
            st.info(f"F = {f:.4f}, P = {p:.4f}")
            interpretation(p)
            sns.boxplot(x=df[cat], y=df[num])
            st.pyplot(plt)

    # --- Two-Way ANOVA ---
    elif test_type == "ANOVA (Two-Way)":
        num = st.sidebar.selectbox("Numeric Variable (Dependent)", numeric_cols)
        factor1 = st.sidebar.selectbox("Factor 1", cat_cols)
        factor2 = st.sidebar.selectbox("Factor 2", [c for c in cat_cols if c != factor1])
        if st.button("Run ANOVA (Two-Way)"):
            formula = f"{num} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.dataframe(anova_table)
            for factor in anova_table.index:
                f = anova_table.loc[factor, "F"]
                p = anova_table.loc[factor, "PR(>F)"]
                st.markdown(f"**{factor} → F = {f:.4f}, P = {p:.4f}**")
                interpretation(p)

    # --- Z-test for Mean ---
    elif test_type == "Z-test for Mean":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        pop_mean = st.sidebar.number_input("Enter Population Mean (μ₀)", value=0.0)
        sample = df[num].dropna()
        x_bar = np.mean(sample)
        s = np.std(sample, ddof=1)
        n = len(sample)
        z = (x_bar - pop_mean) / (s / np.sqrt(n))
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        if st.button("Run Z-test for Mean"):
            st.info(f"Z = {z:.4f}, P = {p:.4f}")
            interpretation(p)
            st.markdown(f"**Sample Mean = {x_bar:.4f}, Population Mean = {pop_mean:.4f}, n = {n}**")

    # --- Z-test for Proportion ---
    elif test_type == "Z-test for Proportion":
        success_count = st.sidebar.number_input("Number of Successes", min_value=0)
        n = st.sidebar.number_input("Sample Size (n)", min_value=1)
        pop_prop = st.sidebar.number_input("Population Proportion (p₀)", min_value=0.0, max_value=1.0, value=0.5)
        if st.button("Run Z-test for Proportion"):
            p_hat = success_count / n
            se = np.sqrt((pop_prop * (1 - pop_prop)) / n)
            z = (p_hat - pop_prop) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            st.info(f"Z = {z:.4f}, P = {p_value:.4f}")
            interpretation(p_value)
            st.markdown(f"**Observed Proportion = {p_hat:.4f}, Expected = {pop_prop:.4f}, n = {n}**")

    # --- Chi-square Test ---
    elif test_type == "Chi-square Test":
        cat1 = st.sidebar.selectbox("Categorical Variable 1", cat_cols)
        cat2 = st.sidebar.selectbox("Categorical Variable 2", [c for c in cat_cols if c != cat1])
        if st.button("Run Chi-square Test"):
            table = pd.crosstab(df[cat1], df[cat2])
            chi2, p, dof, ex = stats.chi2_contingency(table)
            st.success(f"Chi² = {chi2:.3f}, P = {p:.4f}, DF = {dof}")
            interpretation(p)
            st.write("**Expected Frequencies:**")
            st.dataframe(pd.DataFrame(ex, index=table.index, columns=table.columns))
