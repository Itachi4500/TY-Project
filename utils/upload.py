import streamlit as st
import pandas as pd

def upload_data(preview_rows=5):
    """
    Upload and preview a dataset (CSV or Excel).
    
    Parameters:
        preview_rows (int): Number of rows to show after upload.
        
    Returns:
        DataFrame or None
    """
    st.markdown("### 📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # File Info
            st.markdown(f"**Filename:** `{uploaded_file.name}`")
            st.markdown(f"**Size:** `{round(uploaded_file.size / 1024, 2)} KB`")

            # Read File
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Store in session state
            st.session_state.raw_df = df.copy()
            st.session_state.cleaned_df = df.copy()

            st.success("✅ Dataset uploaded and cached successfully!")

            # Preview
            if preview_rows > 0:
                st.markdown(f"#### 👁️ Preview ({preview_rows} rows):")
                st.dataframe(df.head(preview_rows))

            return df

        except Exception as e:
            st.error(f"❌ Failed to upload file: {e}")
            return None

    return None
