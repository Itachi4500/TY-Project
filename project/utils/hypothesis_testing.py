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
    💡 Advanced Smart Hypothesis Testing Suite
    Automatically detects suitable columns, suggests the correct test,
    checks assumptions, runs statistical tests, and interprets results clearly.
    """

    st.title("🧠 AI-Guided Hypothesis Testing Assistant")

    # --- Detect column types ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols and not cat_cols:
        st.error("⚠️ No valid numeric or categorical columns detected in dataset.")
        return

    st.markdown("### 📋 Data Overview")
    st.write(f"**Numeric Columns:** {numeric_cols}")
    st.write(f"**Categorical Columns:** {cat_cols}")
    st.markdown("---")

    # --- Helper Functions ---
    def interpretation(p_value, alpha=0.05):
        """Return standard statistical interpretation"""
        if p_value < alpha:
            st.error(f"❌ Null Hypothesis Rejected (p = {p_value:.4f} < 0.05)")
            st.markdown("**Interpretation:** There is a *statistically significant effect or difference.*")
        else:
            st.success(f"✅ Fail to Reject Null Hypothesis (p = {p_value:.4f} ≥ 0.05)")
            st.markdown("**Interpretation:** No *statistically significant difference/effect* detected.")

    def plot_qq(data, title):
        fig = plt.figure()
        sm.qqplot(data, line='45', fit=True)
        plt.title(title)
        st.pyplot(fig)

    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1)+(ny-1)*np.var(y, ddof=1))/(nx+ny-2))
        return (np.mean(x)-np.mean(y))/pooled_std

    # --- Sidebar: User Selection ---
    test_mode = st.sidebar.radio("Choose Analysis Mode:", ["🧠 Auto Detection", "🧮 Manual Test Selection"])

    # ==================================================================================
    # 🧠 AUTO DETECTION MODE
    # ==================================================================================
    if test_mode == "🧠 Auto Detection":
        st.subheader("🤖 Automatic Test Recommendation")

        # Step 1: Column Detection
        num_col = st.selectbox("Select Numeric Variable (Dependent)", numeric_cols)
        cat_col = st.selectbox("Select Categorical Variable (Independent)", cat_cols if cat_cols else [None])

        if cat_col and cat_col in df.columns:
            groups = df[cat_col].dropna().unique()
            n_groups = len(groups)

            st.write(f"**Groups Detected:** {n_groups} ({groups})")

            # Step 2: Determine Best Test Type
            if n_groups == 2:
                # Check normality for both groups
                g1 = df[df[cat_col] == groups[0]][num_col].dropna()
                g2 = df[df[cat_col] == groups[1]][num_col].dropna()

                p_norm1 = stats.shapiro(g1)[1] if len(g1) >= 3 else 0.0
                p_norm2 = stats.shapiro(g2)[1] if len(g2) >= 3 else 0.0
                lev_p = stats.levene(g1, g2)[1]

                st.markdown("### 📊 Assumption Checks")
                st.write(f"Normality (Shapiro): {groups[0]} → {p_norm1:.4f}, {groups[1]} → {p_norm2:.4f}")
                st.write(f"Equal Variance (Levene): p = {lev_p:.4f}")

                # Visualize distributions
                fig, ax = plt.subplots()
                sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
                plt.title("Group Distribution Comparison")
                st.pyplot(fig)

                # Step 3: Recommend Test
                if p_norm1 < 0.05 or p_norm2 < 0.05:
                    test_recommended = "Mann–Whitney U Test (Non-parametric)"
                elif lev_p < 0.05:
                    test_recommended = "Welch’s T-test (Unequal Variances)"
                else:
                    test_recommended = "Independent T-test"

                st.success(f"✅ Recommended Test: **{test_recommended}**")

                # Step 4: Run Selected Test Automatically
                if st.button("Run Recommended Test"):
                    if "T-test" in test_recommended:
                        equal_var = "Welch" not in test_recommended
                        t, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
                        st.info(f"T = {t:.4f}, P = {p:.4f}, Cohen’s d = {cohen_d(g1, g2):.3f}")
                        interpretation(p)
                    else:
                        u, p = stats.mannwhitneyu(g1, g2)
                        st.info(f"U = {u:.4f}, P = {p:.4f}")
                        interpretation(p)

            elif n_groups >= 3:
                st.success("Recommended Test: **One-Way ANOVA** (3 or more groups detected)")
                if st.button("Run One-Way ANOVA"):
                    model = ols(f"{num_col} ~ C({cat_col})", data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    st.dataframe(anova_table)
                    p = anova_table["PR(>F)"][0]
                    interpretation(p)
                    sns.boxplot(x=df[cat_col], y=df[num_col])
                    st.pyplot(plt)
                    if p < 0.05:
                        st.info("Post-hoc test (Tukey HSD) is recommended to find which groups differ.")

            else:
                st.warning("Not enough distinct groups found for testing.")
        else:
            st.info("Please select both a numeric and a categorical variable.")

    # ==================================================================================
    # 🧮 MANUAL TEST SELECTION
    # ==================================================================================
    else:
        st.subheader("⚙️ Manual Test Selection")
        test_type = st.sidebar.selectbox("Select Statistical Test", [
            "Independent T-test", "Paired T-test", "Welch’s T-test",
            "Mann–Whitney U Test", "ANOVA (One-Way)", "ANOVA (Two-Way)",
            "Z-test for Mean", "Z-test for Proportion", "Chi-square Test"
        ])

        # --- Independent T-test ---
        if test_type == "Independent T-test":
            num = st.selectbox("Numeric Variable", numeric_cols)
            cat = st.selectbox("Grouping Variable (2 groups)", cat_cols)
            if cat not in df.columns:
                st.error("Invalid categorical variable selected.")
                return
            groups = df[cat].dropna().unique()
            if len(groups) != 2:
                st.warning("Requires exactly 2 groups.")
                return
            g1 = df[df[cat] == groups[0]][num].dropna()
            g2 = df[df[cat] == groups[1]][num].dropna()
            if st.button("Run Independent T-test"):
                t, p = stats.ttest_ind(g1, g2)
                st.info(f"T = {t:.4f}, P = {p:.4f}")
                interpretation(p)

        # --- Paired T-test ---
        elif test_type == "Paired T-test":
            x = st.selectbox("Variable 1", numeric_cols)
            y = st.selectbox("Variable 2", [c for c in numeric_cols if c != x])
            if st.button("Run Paired T-test"):
                t, p = stats.ttest_rel(df[x], df[y])
                st.info(f"T = {t:.4f}, P = {p:.4f}")
                interpretation(p)

        # --- ANOVA (One-Way) ---
        elif test_type == "ANOVA (One-Way)":
            num = st.selectbox("Numeric Variable", numeric_cols)
            cat = st.selectbox("Categorical Variable (3+ groups)", cat_cols)
            if st.button("Run ANOVA"):
                model = ols(f"{num} ~ C({cat})", data=df).fit()
                table = sm.stats.anova_lm(model, typ=2)
                st.dataframe(table)
                p = table["PR(>F)"][0]
                interpretation(p)

        # --- ANOVA (Two-Way) ---
        elif test_type == "ANOVA (Two-Way)":
            num = st.selectbox("Dependent Variable", numeric_cols)
            f1 = st.selectbox("Factor 1", cat_cols)
            f2 = st.selectbox("Factor 2", [c for c in cat_cols if c != f1])
            if st.button("Run Two-Way ANOVA"):
                formula = f"{num} ~ C({f1}) + C({f2}) + C({f1}):C({f2})"
                model = ols(formula, data=df).fit()
                table = sm.stats.anova_lm(model, typ=2)
                st.dataframe(table)
                for factor in table.index:
                    p = table.loc[factor, "PR(>F)"]
                    st.markdown(f"**{factor} → p = {p:.4f}**")
                    interpretation(p)

        # --- Z-test for Mean ---
        elif test_type == "Z-test for Mean":
            num = st.selectbox("Numeric Variable", numeric_cols)
            mu_0 = st.number_input("Population Mean (μ₀)", value=0.0)
            sample = df[num].dropna()
            x_bar, s, n = np.mean(sample), np.std(sample, ddof=1), len(sample)
            z = (x_bar - mu_0) / (s / np.sqrt(n))
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            if st.button("Run Z-test for Mean"):
                st.info(f"Z = {z:.4f}, P = {p:.4f}")
                interpretation(p)
                st.markdown(f"**Sample Mean = {x_bar:.4f}, n = {n}**")

        # --- Z-test for Proportion ---
        elif test_type == "Z-test for Proportion":
            x = st.number_input("Number of Successes", min_value=0)
            n = st.number_input("Sample Size", min_value=1)
            p_0 = st.number_input("Population Proportion (p₀)", min_value=0.0, max_value=1.0, value=0.5)
            if st.button("Run Z-test for Proportion"):
                p_hat = x / n
                se = np.sqrt((p_0 * (1 - p_0)) / n)
                z = (p_hat - p_0) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                st.info(f"Z = {z:.4f}, P = {p_value:.4f}")
                interpretation(p_value)

        # --- Chi-square Test ---
        elif test_type == "Chi-square Test":
            cat1 = st.selectbox("Categorical Variable 1", cat_cols)
            cat2 = st.selectbox("Categorical Variable 2", [c for c in cat_cols if c != cat1])
            if st.button("Run Chi-square Test"):
                table = pd.crosstab(df[cat1], df[cat2])
                chi2, p, dof, ex = stats.chi2_contingency(table)
                st.info(f"Chi² = {chi2:.3f}, DF = {dof}, P = {p:.4f}")
                interpretation(p)
                st.write("Expected Frequencies:")
                st.dataframe(pd.DataFrame(ex, index=table.index, columns=table.columns))
