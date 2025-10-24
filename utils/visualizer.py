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

    elif analysis_type == "Hypothesis Testing":
        st.markdown("Perform statistical tests to check hypotheses about the data.")
        
        test_type = st.sidebar.selectbox("Select Test", ["Independent T-test", "ANOVA (One-Way)"])

        if test_type == "Independent T-test":
            st.markdown("#### Independent Samples T-test")
            st.write("Compares the means for two independent groups.")
            
            if len(numeric_cols) > 0 and len(cat_cols) > 0:
                numeric_var = st.selectbox("Select Numeric Variable (to compare)", numeric_cols)
                cat_var = st.selectbox("Select Categorical Variable (with 2 groups)", cat_cols)
                
                if df[cat_var].nunique() == 2:
                    groups = df[cat_var].unique()
                    group1 = df[df[cat_var] == groups[0]][numeric_var].dropna()
                    group2 = df[df[cat_var] == groups[1]][numeric_var].dropna()

                    if st.button("Run T-test"):
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        st.write(f"**Results for {numeric_var} between '{groups[0]}' and '{groups[1]}':**")
                        st.write(f"T-statistic: `{t_stat:.4f}`")
                        st.write(f"P-value: `{p_value:.4f}`")

                        alpha = 0.05
                        if p_value < alpha:
                            st.success(f"The difference in means is statistically significant (p < {alpha}). We reject the null hypothesis.")
                        else:
                            st.info(f"The difference in means is not statistically significant (p >= {alpha}). We fail to reject the null hypothesis.")
                else:
                    st.error(f"The selected categorical variable '{cat_var}' has {df[cat_var].nunique()} unique values. Please choose a variable with exactly 2 groups for a T-test.")
            else:
                st.warning("T-tests require at least one numeric and one categorical column.")

        elif test_type == "ANOVA (One-Way)":
            st.markdown("#### One-Way ANOVA")
            st.write("Compares the means of three or more independent groups.")

            if len(numeric_cols) > 0 and len(cat_cols) > 0:
                numeric_var = st.selectbox("Select Numeric Variable (to compare)", numeric_cols, key="anova_num")
                cat_var = st.selectbox("Select Categorical Variable (with 3+ groups)", cat_cols, key="anova_cat")

                if df[cat_var].nunique() > 2:
                    groups = df[cat_var].unique()
                    samples = [df[df[cat_var] == g][numeric_var].dropna() for g in groups]

                    if st.button("Run ANOVA"):
                        f_stat, p_value = stats.f_oneway(*samples)
                        st.write(f"**Results for {numeric_var} across groups in '{cat_var}':**")
                        st.write(f"F-statistic: `{f_stat:.4f}`")
                        st.write(f"P-value: `{p_value:.4f}`")

                        alpha = 0.05
                        if p_value < alpha:
                            st.success(f"There is a statistically significant difference in means between at least two groups (p < {alpha}). We reject the null hypothesis.")
                        else:
                            st.info(f"There is no statistically significant difference in means between the groups (p >= {alpha}). We fail to reject the null hypothesis.")
                else:
                    st.error(f"The selected categorical variable '{cat_var}' has {df[cat_var].nunique()} unique values. Please choose a variable with 3 or more groups for ANOVA.")
            else:
                st.warning("ANOVA requires at least one numeric and one categorical column.")
