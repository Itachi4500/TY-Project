import streamlit as st
import pandas as pd

def upload_data(preview_rows=5):
    """
    Upload and preview a dataset (CSV or Excel)
    Supports uploads up to 10GB (configured in config.toml)
    """

    st.markdown("### 📁 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            # File info
            size_gb = uploaded_file.size / (1024**3)

            st.markdown(f"**Filename:** `{uploaded_file.name}`")
            st.markdown(f"**Size:** `{size_gb:.2f} GB`")

            # Read dataset
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Store session
            st.session_state.raw_df = df.copy()
            st.session_state.cleaned_df = df.copy()

            st.success("✅ Dataset uploaded and cached successfully!")

            if preview_rows > 0:
                st.markdown(f"#### 👁️ Preview ({preview_rows} rows)")
                st.dataframe(df.head(preview_rows))

            return df

        except Exception as e:
            st.error(f"❌ Failed to upload file: {str(e)}")
            return None

    return None
