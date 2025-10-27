import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def show_visuals(df):
    """
    An advanced Streamlit dashboard for data visualization and statistical analysis.
    """
    st.title("📊 Advanced Visualization & Analysis Dashboard")

    # --- Column Detection ---
    # Convert object columns that can be parsed as datetime
    
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            continue # If conversion fails, keep it as object

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()

    # --- Sidebar Navigation ---
    
    with st.sidebar:
        st.header("🧭 Visualization & Analysis Navigator")
        analysis_type = st.selectbox("Choose Analysis Type", [
            "Histogram", "Heatmap", "Bar Chart", "Pie Chart", "Donut Chart",
            "Line Chart", "Scatter Plot", "Bubble Chart", "Pair Plot", "Hypothesis Testing"
        ])
        st.markdown("---")

    st.subheader(f"🗂️ {analysis_type}")

    # --- Chart Implementations ---

    if analysis_type == "Histogram":
        st.markdown("Shows the distribution of a single numeric variable.")
        if numeric_cols:
            col = st.sidebar.selectbox("Select Numeric Column", numeric_cols, key="hist_col")
            bins = st.sidebar.slider("Number of Bins", 5, 150, 30, key="hist_bins")
            color_group = st.sidebar.selectbox("Group By (Color)", ["None"] + cat_cols, key="hist_color")
            
            histnorm = st.sidebar.selectbox("Normalization", [None, 'percent', 'probability density'], key="hist_norm")
            marginal = st.sidebar.selectbox("Marginal Plot", [None, 'box', 'violin', 'rug'], key="hist_marginal")

            fig = px.histogram(
                df,
                x=col,
                nbins=bins,
                color=color_group if color_group != "None" else None,
                marginal=marginal,
                histnorm=histnorm,
                barmode='overlay',
                opacity=0.7,
                title=f"Distribution of {col}"
            )
            fig.update_layout(xaxis_title=col, yaxis_title="Count" if histnorm is None else histnorm.title())
            st.plotly_chart(fig, use_container_width=True)

            if st.sidebar.checkbox("Show Statistics", value=True, key="hist_stats"):
                st.markdown("#### 📌 Descriptive Statistics")
                st.dataframe(df[[col]].describe().T)
        else:
            st.warning("No numeric columns available for a histogram.")

    elif analysis_type == "Heatmap":
        st.markdown("Visualizes the correlation matrix of numeric variables.")
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5, ax=ax)
            plt.title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("At least two numeric columns are required for a heatmap.")

    elif analysis_type == "Bar Chart":
        st.markdown("Compares a numeric value across different categories.")
        if cat_cols:
            x_col = st.sidebar.selectbox("Categorical Column (X-axis)", cat_cols, key="bar_x")
            y_col = st.sidebar.selectbox("Numeric Column (Y-axis)", numeric_cols, key="bar_y")
            agg_func = st.sidebar.selectbox("Aggregation Function", ["Mean", "Sum", "Count"], key="bar_agg")
            
            if agg_func == "Mean":
                grouped_data = df.groupby(x_col)[y_col].mean().reset_index()
            elif agg_func == "Sum":
                grouped_data = df.groupby(x_col)[y_col].sum().reset_index()
            else: # Count
                grouped_data = df[x_col].value_counts().reset_index()
                grouped_data.columns = [x_col, 'Count']
                y_col = 'Count'

            fig = px.bar(grouped_data.sort_values(by=y_col, ascending=False), x=x_col, y=y_col, title=f"{agg_func} of {y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("A categorical column is required for a bar chart.")

    elif analysis_type in ["Pie Chart", "Donut Chart"]:
        st.markdown("Shows the proportion of categories in a dataset.")
        if cat_cols:
            col = st.sidebar.selectbox("Select Categorical Column", cat_cols, key="pie_col")
            value_counts = df[col].value_counts()
            top_n = st.sidebar.slider("Number of Categories to Show", 3, min(20, len(value_counts)), 10, key="pie_n")
            
            top_data = value_counts.nlargest(top_n)
            if len(value_counts) > top_n:
                top_data['Others'] = value_counts.iloc[top_n:].sum()

            fig = px.pie(
                names=top_data.index,
                values=top_data.values,
                hole=0.4 if analysis_type == "Donut Chart" else 0.0,
                title=f"Proportion of {col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("A categorical column is required for pie/donut charts.")

    elif analysis_type == "Line Chart":
        st.markdown("Displays data points connected by straight line segments, ideal for time-series data.")
        if not numeric_cols:
            st.warning("At least one numeric column is required for a line chart.")
        else:
            x_axis_options = datetime_cols + numeric_cols + cat_cols
            x_col = st.sidebar.selectbox("X-Axis", x_axis_options, key="line_x")
            y_col = st.sidebar.selectbox("Y-Axis (Numeric)", numeric_cols, key="line_y")
            color_group = st.sidebar.selectbox("Group By (Color)", ["None"] + cat_cols, key="line_color")
            show_markers = st.sidebar.checkbox("Show Markers", value=True, key="line_markers")

            temp_df = df.copy()
            
            # Time series resampling
            if x_col in datetime_cols:
                resample_rule = st.sidebar.selectbox("Resample Time Series", ["None", "Day (D)", "Week (W)", "Month (M)", "Quarter (Q)", "Year (Y)"], key="line_resample")
                if resample_rule != "None":
                    rule_code = resample_rule.split('(')[1][0]
                    agg_func = st.sidebar.selectbox("Aggregation for Resampling", ["mean", "sum", "median"], key="line_resample_agg")
                    temp_df = df.set_index(x_col).groupby(color_group if color_group != "None" else lambda x: True)[y_col].resample(rule_code).agg(agg_func).reset_index()


            fig = px.line(
                temp_df.sort_values(by=x_col), 
                x=x_col, 
                y=y_col, 
                color=color_group if color_group != "None" else None,
                markers=show_markers,
                title=f"{y_col} over {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Scatter Plot":
        st.markdown("Examines the relationship between two numeric variables.")
        if len(numeric_cols) >= 2:
            x = st.sidebar.selectbox("X-Axis", numeric_cols, key="scatter_x")
            y = st.sidebar.selectbox("Y-Axis", [col for col in numeric_cols if col != x], key="scatter_y")
            color = st.sidebar.selectbox("Color By", ["None"] + cat_cols, key="scatter_color")
            fig = px.scatter(df, x=x, y=y, color=color if color != "None" else None, title=f"Scatter Plot of {y} vs {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("At least two numeric columns are required for a scatter plot.")

    elif analysis_type == "Bubble Chart":
        st.markdown("A scatter plot where a third dimension is added by the size of the markers.")
        if len(numeric_cols) >= 3:
            x = st.sidebar.selectbox("X-Axis", numeric_cols, key="bubble_x")
            y = st.sidebar.selectbox("Y-Axis", [col for col in numeric_cols if col != x], key="bubble_y")
            size = st.sidebar.selectbox("Bubble Size", [col for col in numeric_cols if col not in [x, y]], key="bubble_size")
            color = st.sidebar.selectbox("Color By", ["None"] + cat_cols, key="bubble_color")
            fig = px.scatter(df, x=x, y=y, size=size, color=color if color != "None" else None, title=f"Bubble Chart of {y} vs {x} (size by {size})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("At least three numeric columns are required for a bubble chart.")

    elif analysis_type == "Pair Plot":
        st.markdown("Visualizes relationships between multiple variables at once.")
        if len(numeric_cols) >= 2:
            st.info("Pair plots can be slow to render for datasets with many columns.")
            selected_vars = st.sidebar.multiselect("Select variables for pair plot", numeric_cols, default=numeric_cols[:min(len(numeric_cols), 5)])
            hue_col = st.sidebar.selectbox("Color By (Hue)", ["None"] + cat_cols, key="pair_hue")
            
            if len(selected_vars) > 1:
                if st.button("Generate Pair Plot"):
                    with st.spinner("Generating plot..."):
                        fig = sns.pairplot(df[selected_vars + ([hue_col] if hue_col != "None" else [])], hue=hue_col if hue_col != "None" else None)
                        st.pyplot(fig)
            else:
                st.warning("Please select at least two variables.")
        else:
            st.warning("At least two numeric columns are required for a pair plot.")

    # ... (rest of your code remains unchanged above)

    elif analysis_type == "Hypothesis Testing":
        st.markdown("Perform a variety of statistical tests and view assumptions, results, and visualizations.")

        st.sidebar.markdown("### Test Configuration")
        test_type = st.sidebar.selectbox("Select Test", [
            "Independent T-test",
            "Paired T-test",
            "Welch's T-test (Unequal Variance)",
            "Mann–Whitney U Test (Non-parametric)",
            "ANOVA (One-Way)",
            "Chi-square Test",
        ])

        # Helper for variable selection
        def get_var_select(numeric_req=1, cat_req=1, pair_req=False, unique_min=None, unique_max=None):
            num = st.sidebar.selectbox("Numeric Variable", numeric_cols) if numeric_req else None
            cat = st.sidebar.selectbox("Categorical Variable", cat_cols) if cat_req else None
            pair = None
            if pair_req:
                pair = st.sidebar.selectbox("Paired Numeric Variable", [col for col in numeric_cols if col != num])
            # Unique group checks
            if cat and unique_min:
                n_unique = df[cat].nunique()
                if n_unique < unique_min:
                    st.error(f"Categorical variable must have at least {unique_min} unique values.")
                    return None, None, None
            if cat and unique_max:
                n_unique = df[cat].nunique()
                if n_unique > unique_max:
                    st.error(f"Categorical variable must have at most {unique_max} unique values.")
                    return None, None, None
            return num, cat, pair

        # Effect size calculator
        def cohen_d(x, y):
            nx, ny = len(x), len(y)
            pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx + ny - 2))
            return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std else 0

        if test_type == "Independent T-test":
            st.markdown("#### Independent Samples T-test")
            num, cat, _ = get_var_select(numeric_req=1, cat_req=1, unique_min=2, unique_max=2)
            if num and cat:
                groups = df[cat].unique()
                group1 = df[df[cat] == groups[0]][num].dropna()
                group2 = df[df[cat] == groups[1]][num].dropna()
                st.write(f"Comparing `{num}` between {groups[0]} and {groups[1]}")
                st.pyplot(sns.boxplot(x=df[cat], y=df[num]))
                if st.button("Run T-test"):
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    st.write(f"T-statistic: `{t_stat:.4f}` | P-value: `{p_value:.4f}`")
                    st.write(f"Cohen's d: `{cohen_d(group1, group2):.4f}`")
                    st.write("Sample sizes:", len(group1), len(group2))
                    st.write(f"Means: {np.mean(group1):.2f}, {np.mean(group2):.2f}")
                    st.write(f"Std dev: {np.std(group1, ddof=1):.2f}, {np.std(group2, ddof=1):.2f}")

        elif test_type == "Paired T-test":
            st.markdown("#### Paired Samples T-test")
            num1 = st.sidebar.selectbox("Numeric Variable 1", numeric_cols)
            num2 = st.sidebar.selectbox("Numeric Variable 2 (paired)", [col for col in numeric_cols if col != num1])
            st.write(f"Comparing paired samples: `{num1}` vs `{num2}`")
            df_paired = df[[num1, num2]].dropna()
            st.pyplot(sns.boxplot(data=df_paired))
            if st.button("Run Paired T-test"):
                t_stat, p_value = stats.ttest_rel(df_paired[num1], df_paired[num2])
                st.write(f"T-statistic: `{t_stat:.4f}` | P-value: `{p_value:.4f}`")
                st.write(f"Mean difference: {(df_paired[num1] - df_paired[num2]).mean():.4f}")

        elif test_type == "Welch's T-test (Unequal Variance)":
            st.markdown("#### Welch's T-test")
            num, cat, _ = get_var_select(numeric_req=1, cat_req=1, unique_min=2, unique_max=2)
            if num and cat:
                groups = df[cat].unique()
                group1 = df[df[cat] == groups[0]][num].dropna()
                group2 = df[df[cat] == groups[1]][num].dropna()
                st.pyplot(sns.boxplot(x=df[cat], y=df[num]))
                if st.button("Run Welch's T-test"):
                    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                    st.write(f"Welch's T-statistic: `{t_stat:.4f}` | P-value: `{p_value:.4f}`")
                    st.write(f"Cohen's d: `{cohen_d(group1, group2):.4f}`")

        elif test_type == "Mann–Whitney U Test (Non-parametric)":
            st.markdown("#### Mann–Whitney U Test")
            num, cat, _ = get_var_select(numeric_req=1, cat_req=1, unique_min=2, unique_max=2)
            if num and cat:
                groups = df[cat].unique()
                group1 = df[df[cat] == groups[0]][num].dropna()
                group2 = df[df[cat] == groups[1]][num].dropna()
                st.pyplot(sns.violinplot(x=df[cat], y=df[num]))
                if st.button("Run Mann–Whitney U Test"):
                    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    st.write(f"Mann–Whitney U statistic: `{u_stat:.4f}` | P-value: `{p_value:.4f}`")
                    st.write(f"Median: {np.median(group1):.2f}, {np.median(group2):.2f}")

        elif test_type == "ANOVA (One-Way)":
            st.markdown("#### One-Way ANOVA")
            num, cat, _ = get_var_select(numeric_req=1, cat_req=1, unique_min=3)
            if num and cat:
                groups = df[cat].unique()
                samples = [df[df[cat] == g][num].dropna() for g in groups]
                st.pyplot(sns.boxplot(x=df[cat], y=df[num]))
                if st.button("Run ANOVA"):
                    f_stat, p_value = stats.f_oneway(*samples)
                    st.write(f"F-statistic: `{f_stat:.4f}` | P-value: `{p_value:.4f}`")
                    # Effect size eta squared
                    grand_mean = df[num].mean()
                    ss_between = sum([len(s)*((s.mean()-grand_mean)**2) for s in samples])
                    ss_total = sum((df[num]-grand_mean)**2)
                    eta_sq = ss_between / ss_total if ss_total else 0
                    st.write(f"Eta squared: `{eta_sq:.4f}`")

        elif test_type == "Chi-square Test":
            st.markdown("#### Chi-square Test of Independence")
            cat1 = st.sidebar.selectbox("Categorical Variable 1", cat_cols)
            cat2 = st.sidebar.selectbox("Categorical Variable 2", [col for col in cat_cols if col != cat1])
            table = pd.crosstab(df[cat1], df[cat2])
            st.write("Contingency Table:")
            st.dataframe(table)
            if st.button("Run Chi-square Test"):
                chi2, p, dof, ex = stats.chi2_contingency(table)
                st.write(f"Chi-square statistic: `{chi2:.4f}` | P-value: `{p:.4f}` | df: `{dof}`")
                st.write("Expected frequencies:")
                st.dataframe(pd.DataFrame(ex, index=table.index, columns=table.columns))

        # Multiple comparison corrections
        if st.sidebar.checkbox("Show correction options for multiple tests", value=False):
            st.sidebar.markdown("#### Multiple Testing Correction")
            correction = st.sidebar.selectbox("Correction Method", [
                "None", "Bonferroni", "Holm", "Benjamini/Hochberg"
            ])
            st.info(f"Selected: {correction} (not applied automatically, see scipy.stats/multipletests for implementation)")

        # Show test assumptions
        if st.sidebar.checkbox("Show test assumptions and checks", value=True):
            st.markdown("#### Test Assumptions & Checks")
            st.write("Check normality, sample size, variance, etc. for your selected variables.")
            # Example: Normality check for selected numeric variable
            if 'num' in locals() and num:
                st.write("Normality check (Shapiro-Wilk):")
                stat, p = stats.shapiro(df[num].dropna())
                st.write(f"Statistic: `{stat:.4f}` | P-value: `{p:.4f}`")
                if p < 0.05:
                    st.warning("Data is likely **not normal** (p < 0.05). Consider non-parametric tests.")
                else:
                    st.success("Data is likely **normal** (p >= 0.05).")

            # Example: Variance check for two groups (Levene's Test)
            if test_type in ["Independent T-test", "Welch's T-test (Unequal Variance)", "Mann–Whitney U Test (Non-parametric)"] and 'group1' in locals() and 'group2' in locals():
                stat, p = stats.levene(group1, group2)
                st.write("Levene's Test for equal variances:")
                st.write(f"Statistic: `{stat:.4f}` | P-value: `{p:.4f}`")
                if p < 0.05:
                    st.warning("Groups likely have **unequal variances** (p < 0.05).")
                else:
                    st.success("Groups likely have **equal variances** (p >= 0.05).")

        st.markdown("---")
        st.write("**Tip:** For more advanced tests, see [statsmodels](https://www.statsmodels.org/) or [scipy.stats docs](https://docs.scipy.org/doc/scipy/reference/stats.html).")
