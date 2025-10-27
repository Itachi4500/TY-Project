import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import plotly.express as px # Added import for plotting

def clean_data(df):
    """
    A Streamlit function to interactively clean a pandas DataFrame.
    """
    st.subheader("🧹 Data Cleaning Dashboard")

    # Initial check
    if df.empty:
        st.error("❌ The uploaded dataset is empty.")
        return df

    st.markdown("---")
    
    # 1. Handle Missing Values
    st.markdown("### 🔍 1. Handle Missing Values")
    missing_values_count = df.isnull().sum()
    st.write("Missing Values Count:")
    st.dataframe(missing_values_count[missing_values_count > 0])
    
    missing_action = st.radio("Choose how to handle missing values:", ["None", "Fill NA", "Drop NA"], key="missing_handler")

    if missing_action == "Fill NA":
        method = st.selectbox("Select Fill Method:", ["Forward Fill", "Backward Fill", "Mean", "Median", "Mode", "Zero"])
        # Separate numeric and categorical columns for filling
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns
        
        if method == "Forward Fill":
            df = df.fillna(method='ffill')
        elif method == "Backward Fill":
            df = df.fillna(method='bfill')
        elif method == "Mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == "Median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == "Mode":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif method == "Zero":
            df = df.fillna(0)
        st.success(f"✅ Missing values filled using: **{method}**")

    elif missing_action == "Drop NA":
        original_size = df.shape[0]
        df = df.dropna()
        dropped = original_size - df.shape[0]
        st.success(f"✅ Dropped **{dropped}** rows with missing values.")

    st.write("Remaining Missing Values:")
    st.dataframe(df.isnull().sum())

    st.markdown("---")

    # 2. Outlier Detection and Removal (MODIFIED SECTION)
    st.markdown("### 📊 2. Outlier Detection & Removal")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        outlier_col = st.selectbox(
            "Select a column to visualize and remove outliers from:",
            numeric_cols,
            key="outlier_col"
        )

        # --- NEW: Interactive Box Plot ---
        if outlier_col:
            st.write(f"**Distribution of `{outlier_col}`**")
            fig = px.box(df, y=outlier_col, title=f"Box Plot for '{outlier_col}'")
            st.plotly_chart(fig, use_container_width=True)

        method = st.selectbox(
            "Select Outlier Removal Method:",
            ["None", "IQR Method", "Z-Score Method"],
            key="outlier_method"
        )

        if method != "None":
            if st.button(f"Remove Outliers using {method}", key="remove_outliers_btn"):
                original_size = df.shape[0]
                
                if method == "IQR Method":
                    Q1 = df[outlier_col].quantile(0.25)
                    Q3 = df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                
                elif method == "Z-Score Method":
                    # Handle potential NaN values before calculating Z-score
                    col_data = df[outlier_col].dropna()
                    z_scores = np.abs(zscore(col_data))
                    # Filter based on the indices of the non-outlier data
                    df = df.loc[col_data.index[z_scores < 3]]

                new_size = df.shape[0]
                st.success(f"✅ **{original_size - new_size}** outliers removed. Rows reduced from {original_size} to {new_size}.")
    else:
        st.warning("⚠️ No numeric columns available for outlier detection.")

    if df.empty:
        st.error("❌ DataFrame is empty after cleaning. Please check your steps or upload a new dataset.")
        return df

    st.markdown("---")

    # 3. Column Filtering (To select columns to keep)
    st.markdown("### 🧠 3. Column Selection for Analysis")
    all_columns = df.columns.tolist()
    selected_cols = st.multiselect("Select columns to keep in the final dataset:", all_columns, default=all_columns)
    df = df[selected_cols]
    st.write("Filtered DataFrame Preview:")
    st.dataframe(df.head())

    st.markdown("---")
    
    # 4. Save and Export
    st.markdown("### ✅ 4. Save and Export Cleaned Data")
    if st.button("Save Cleaned Dataset to Session"):
        st.session_state.cleaned_df = df.copy()
        st.success("Cleaned dataset saved! You can now use it in other parts of the app.")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Data as CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    return df
