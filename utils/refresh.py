import streamlit as st
import pandas as pd

def refresh_data(callback=None, preview_rows=5):
    """
    Refresh the cleaned dataset to its original uploaded state.

    Parameters:
        callback (function): Optional function to call after refresh.
        preview_rows (int): Number of rows to display after refreshing.
    """
    st.markdown("### 🔄 Refresh Dataset to Original")

    if "raw_df" in st.session_state and isinstance(st.session_state.raw_df, pd.DataFrame):
        if st.button("🔁 Reset to Uploaded Dataset"):
            st.session_state.cleaned_df = st.session_state.raw_df.copy()
            st.success("✅ Dataset refreshed to original version.")
            
            if preview_rows > 0:
                st.markdown(f"#### 👁️ Preview ({preview_rows} rows):")
                st.dataframe(st.session_state.cleaned_df.head(preview_rows))

            if callback:
                callback()

    else:
        st.warning("⚠️ No uploaded dataset found in memory. Please upload a file first.")
