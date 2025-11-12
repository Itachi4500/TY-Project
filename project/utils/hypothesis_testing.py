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
    AI-Powered Smart Hypothesis Testing Suite
    with automatic test recommendations, assumption checks,
    and visual + statistical interpretation.
    """
    st.title("🧠 AI-Powered Hypothesis Testing Suite")

    # --- Detect Data Types ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ No numeric columns found in your dataset.")
        return

    # --- Sidebar Controls ---
    test_type = st.sidebar.selectbox("Select Statistical Test", [
        "Automatic Test Recommendation 🤖",
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
        """Provides hypothesis interpretation."""
        if p_value < alpha:
            st.error(f"❌ Null Hypothesis Rejected (p = {p_value:.4f} < 0.05)")
            st.markdown("**Interpretation:** There is a *statistically significant difference or effect.*")
        else:
            st.success(f"✅ Fail to Reject Null Hypothesis (p = {p_value:.4f} ≥ 0.05)")
            st.markdown("**Interpretation:** No *statistically significant difference or effect* detected.")

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

    # --- 🔮 Automatic Test Recommendation ---
    if test_type == "Automatic Test Recommendation 🤖":
        st.subheader("🤖 Automatic Statistical Test Recommender")

        num_cols = st.multiselect("Select Numeric Variable(s)", numeric_cols)
        cat_cols_selected = st.multiselect("Select Categorical Variable(s)", cat_cols)

        if len(num_cols) == 1 and len(cat_cols_selected) == 1:
            num = num_cols[0]
            cat = cat_cols_selected[0]
            unique_groups = df[cat].dropna().unique()
            st.write(f"📊 Selected variable: `{num}` grouped by `{cat}` ({len(unique_groups)} groups)")

            # Normality Test
            norm_results = []
            for g in unique_groups:
                p_norm = stats.shapiro(df[df[cat] == g][num].dropna())[1]
                norm_results.append(p_norm)
            normal = all(p >= 0.05 for p in norm_results)

            # Variance Equality
            groups = [df[df[cat] == g][num].dropna() for g in unique_groups]
            p_var = stats.levene(*groups)[1]
            equal_var = p_var >= 0.05

            st.markdown("### 🔍 Assumption Check Summary")
            st.write(f"Normality across groups: {'✅ Normal' if normal else '❌ Not Normal'}")
            st.write(f"Equal Variances: {'✅ Yes' if equal_var else '❌ No'}")

            # Recommendation Engine
            st.markdown("### 🧠 Recommended Test")
            if len(unique_groups) == 2:
                if normal and equal_var:
                    st.success("Recommended Test: **Independent T-test** ✅")
                elif normal and not equal_var:
                    st.info("Recommended Test: **Welch’s T-test** (unequal variances)")
                else:
                    st.warning("Recommended Test: **Mann–Whitney U Test** (non-parametric)")
            elif len(unique_groups) > 2:
                if normal:
                    st.success("Recommended Test: **One-Way ANOVA** ✅")
                else:
                    st.warning("Recommended Test: **Kruskal-Wallis H Test** (non-parametric)")

        elif len(num_cols) == 2 and len(cat_cols_selected) == 0:
            st.info("Recommended Test: **Paired T-test** (same subjects before/after)")
        elif len(cat_cols_selected) == 2 and len(num_cols) == 0:
            st.info("Recommended Test: **Chi-square Test of Independence**")
        else:
            st.warning("Please select appropriate variables for meaningful recommendation.")

        st.markdown("---")

    # --- Independent T-test ---
    elif test_type == "Independent T-test":
        if not cat_cols:
            st.warning("No categorical variable found.")
            return
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Grouping Variable (2 groups)", cat_cols)
        if cat not in df.columns:
            st.error(f"Column '{cat}' not found in dataset.")
            return

        groups = df[cat].dropna().unique()
        if len(groups) != 2:
            st.warning("Categorical variable must have exactly 2 unique values.")
            return

        g1 = df[df[cat] == groups[0]][num].dropna()
        g2 = df[df[cat] == groups[1]][num].dropna()

        st.pyplot(sns.boxplot(x=df[cat], y=df[num]))
        stat1, p1 = stats.shapiro(g1)
        stat2, p2 = stats.shapiro(g2)
        st.write(f"{groups[0]} → p = {p1:.4f}, {groups[1]} → p = {p2:.4f}")
        lev_stat, lev_p = stats.levene(g1, g2)
        st.write(f"Levene’s p = {lev_p:.4f}")
        if st.button("Run Independent T-test"):
            t, p = stats.ttest_ind(g1, g2)
            st.info(f"T = {t:.4f}, P = {p:.4f}, Cohen’s d = {cohen_d(g1, g2):.3f}")
            interpretation(p)

    # --- Paired T-test ---
    elif test_type == "Paired T-test":
        x = st.sidebar.selectbox("Variable 1", numeric_cols)
        y = st.sidebar.selectbox("Variable 2 (paired)", [c for c in numeric_cols if c != x])
        diff = df[x] - df[y]
        if st.button("Run Paired T-test"):
            stat, p = stats.ttest_rel(df[x], df[y])
            st.info(f"T = {stat:.4f}, P = {p:.4f}")
            interpretation(p)

    # --- Welch’s T-test ---
    elif test_type == "Welch’s T-test":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        cat = st.sidebar.selectbox("Grouping Variable (2 groups)", cat_cols)
        if cat not in df.columns:
            st.error(f"Column '{cat}' not found in dataset.")
            return
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
        if cat not in df.columns:
            st.error(f"Column '{cat}' not found.")
            return
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
            p = anova_table["PR(>F)"][0]
            interpretation(p)
            sns.boxplot(x=df[cat], y=df[num])
            st.pyplot(plt)

    # --- Two-Way ANOVA ---
    elif test_type == "ANOVA (Two-Way)":
        num = st.sidebar.selectbox("Dependent Numeric Variable", numeric_cols)
        factor1 = st.sidebar.selectbox("Factor 1", cat_cols)
        factor2 = st.sidebar.selectbox("Factor 2", [c for c in cat_cols if c != factor1])
        if st.button("Run ANOVA (Two-Way)"):
            formula = f"{num} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.dataframe(anova_table)
            for factor in anova_table.index:
                p = anova_table.loc[factor, "PR(>F)"]
                interpretation(p)

    # --- Z-test for Mean ---
    elif test_type == "Z-test for Mean":
        num = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        pop_mean = st.sidebar.number_input("Population Mean (μ₀)", value=0.0)
        sample = df[num].dropna()
        x_bar, s, n = np.mean(sample), np.std(sample, ddof=1), len(sample)
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
            st.markdown(f"**Observed = {p_hat:.4f}, Expected = {pop_prop:.4f}, n = {n}**")

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
