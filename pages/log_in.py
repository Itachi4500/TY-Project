import streamlit as st
import os
from datetime import datetime

LOG_DIR = "logs"

# Page Config
st.set_page_config(page_title="Logs | Data Analysis Assistant", layout="wide")

st.title("📋 Application Logs")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Get list of available log files
log_files = sorted(os.listdir(LOG_DIR), reverse=True)

if not log_files:
    st.warning("No logs found yet!")
else:
    # Select log file
    selected_log = st.selectbox("Choose a log file to view:", log_files)

    log_path = os.path.join(LOG_DIR, selected_log)

    st.subheader(f"📄 Viewing: `{selected_log}`")

    # Load log content
    with open(log_path, "r") as f:
        log_content = f.readlines()

    # Filter by log level
    log_level = st.selectbox(
        "Filter by log level:",
        ["ALL", "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
    )

    # Apply filter
    if log_level != "ALL":
        log_content = [line for line in log_content if f"{log_level}" in line]

    # Display logs
    st.text_area(
        "Log Output:",
        "".join(log_content),
        height=450
    )

    # Download button
    with open(log_path, "rb") as fp:
        st.download_button(
            label="⬇ Download Log File",
            data=fp,
            file_name=selected_log,
            mime="text/plain"
        )

    # Refresh logs button
    if st.button("🔄 Refresh Logs"):
        st.rerun()
