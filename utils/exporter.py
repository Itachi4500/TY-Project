import streamlit as st
import pandas as pd

def export_data(df):
    st.subheader("📤 Export Cleaned Dataset")

    if df.empty:
        st.error("❌ No data available to export.")
        return

    file_format = st.selectbox("Choose Export Format", ["CSV", "Excel"])
    filename = st.text_input("Enter file name (without extension)", value="cleaned_data")

    if file_format == "CSV":
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{filename}.csv",
            mime="text/csv"
        )
    else:
        excel_buffer = pd.ExcelWriter(f"{filename}.xlsx", engine="xlsxwriter")
        df.to_excel(excel_buffer, index=False, sheet_name="Sheet1")
        excel_buffer.close()

        with open(f"{filename}.xlsx", "rb") as f:
            st.download_button(
                label="Download Excel",
                data=f,
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
