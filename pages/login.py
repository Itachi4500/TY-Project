import streamlit as st

st.set_page_config(page_title="Login - InsideBox", layout="wide")

# CSS Styling
st.markdown("""
<style>
:root {
    --color-text: #134252;
    --color-border: rgba(94, 82, 64, 0.2);
    --color-primary: #3b82f6;
    --color-primary-hover: #2563eb;
    --radius-base: 8px;
}

.container { max-width: 400px; margin: 80px auto; }
label { font-size: 14px; font-weight: 500; color: var(--color-text); }
input {
    width:100%; padding:12px; margin-top:4px;
    border-radius: var(--radius-base);
    border:1px solid var(--color-border);
}
button.primary {
    width:100%; padding:12px; margin-top:10px;
    background:var(--color-primary);
    color:white; border:none;
    border-radius:var(--radius-base);
    cursor:pointer; font-weight:600;
}
button.primary:hover { background:var(--color-primary-hover);}
.footer-text { margin-top:15px; font-size:14px; }
</style>
""", unsafe_allow_html=True)

# Session Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# UI
st.markdown("<div class='container'>", unsafe_allow_html=True)
st.title("Welcome Back")
st.write("Log in to continue 🚀")

with st.form("login_form"):
    email = st.text_input("Email", placeholder="you@example.com")
    password = st.text_input("Password", type="password", placeholder="••••••••")

    submitted = st.form_submit_button("Log In")

    if submitted:
        if email == "admin@example.com" and password == "admin123":
            st.success("✅ Login Successful!")
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error("❌ Incorrect email or password")

if st.session_state.authenticated:
    st.success(f"You're logged in as: {st.session_state.user_email}")

st.markdown("<p class='footer-text'>Don't have an account? <a href='/signup'>Sign up</a></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
