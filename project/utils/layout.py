import streamlit as st

def select_dashboard_size():
    st.subheader("🖥️ Dashboard Size Configuration")

    size_option = st.selectbox("Select Dashboard Size", [
        "Default (1280x720)", 
        "4:3 (960x720)", 
        "Letter (816x1056)", 
        "Custom"
    ])

    if size_option == "Default (1280x720)":
        width, height = 1280, 720
    elif size_option == "4:3 (960x720)":
        width, height = 960, 720
    elif size_option == "Letter (816x1056)":
        width, height = 816, 1056
    elif size_option == "Custom":
        st.markdown("#### ✏️ Enter Custom Dimensions")
        width = st.number_input("Width (px)", min_value=500, max_value=3000, value=1280, step=10)
        height = st.number_input("Height (px)", min_value=500, max_value=3000, value=720, step=10)

    st.info(f"📐 Selected Dashboard Size: **{int(width)} x {int(height)}** pixels")
    return int(width), int(height)
