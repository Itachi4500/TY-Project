import streamlit as st
from utils.auth import generate_jwt, login_user

def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            token = generate_jwt(username, "client")
            login_user(token)
            st.success("✅ Logged in successfully!")
            st.rerun()
        else:
            st.error("⚠️ Please enter username and password")
