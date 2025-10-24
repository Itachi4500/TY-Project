import streamlit as st
import pandas as pd

def run_eda(df):
    st.subheader("📊 Exploratory Data Analysis")

    if df.empty:
        st.error("❌ The uploaded dataset is empty.")
        return

    analysis_option = st.selectbox("Choose EDA Operation", [
        "Data Overview",
        "Column Types",
        "Summary Statistics",
        "Missing Values",
        "Unique Values",
        "Correlation Matrix",
        "Value Counts for Categorical Columns",
        "Skewness and Kurtosis",
        "Top & Bottom Records"
    ])

    if analysis_option == "Data Overview":
        st.write("### Dataset Preview")
        st.dataframe(df.head())

    elif analysis_option == "Column Types":
        st.write("### Column Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    elif analysis_option == "Summary Statistics":
        st.write("### Summary Statistics")
        st.dataframe(df.describe(include='all').transpose())

    elif analysis_option == "Missing Values":
        st.write("### Missing Values Count")
        missing_count = df.isnull().sum()
        st.dataframe(missing_count[missing_count > 0])

        st.write("### Percentage of Missing Values")
        missing_percent = (missing_count / len(df)) * 100
        st.dataframe(missing_percent[missing_percent > 0])

    elif analysis_option == "Unique Values":
        st.write("### Unique Values in Each Column")
        unique_vals = pd.DataFrame({"Column": df.columns, "Unique Values": df.nunique().values})
        st.dataframe(unique_vals)

    elif analysis_option == "Correlation Matrix":
        st.write("### Correlation Matrix")
        corr_matrix = df.corr(numeric_only=True)
        if not corr_matrix.empty:
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
        else:
            st.warning("No numeric data available for correlation.")

    elif analysis_option == "Value Counts for Categorical Columns":
        cat_cols = df.select_dtypes(include='object').columns
        if len(cat_cols) > 0:
            selected_col = st.selectbox("Select Categorical Column", cat_cols)
            st.write(f"### Value Counts for `{selected_col}`")
            st.dataframe(df[selected_col].value_counts().reset_index().rename(columns={"index": selected_col, selected_col: "Count"}))
        else:
            st.warning("No categorical columns found.")

    elif analysis_option == "Skewness and Kurtosis":
        st.write("### Skewness & Kurtosis (Numeric Columns)")
        numeric_cols = df.select_dtypes(include=['int64', 'float64'])
        if not numeric_cols.empty:
            skew_kurt = pd.DataFrame({
                "Skewness": numeric_cols.skew(),
                "Kurtosis": numeric_cols.kurt()
            })
            st.dataframe(skew_kurt)
        else:
            st.warning("No numeric columns found.")

    elif analysis_option == "Top & Bottom Records":
        n = st.slider("Select number of rows", 1, 20, 5)
        st.write("### Top Records")
        st.dataframe(df.head(n))
        st.write("### Bottom Records")
        st.dataframe(df.tail(n))
