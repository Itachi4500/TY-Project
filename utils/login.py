# utils/logger.py
import streamlit as st

def login_page():
    st.set_page_config(
        page_title="InsideBox - Sign Up",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for styling - mimicking the image's appearance
    st.markdown("""
        <style>
            .stApp {
                background-color: white;
            }
            .main-content {
                display: flex;
                flex-direction: row;
                height: 100vh;
            }
            .left-panel {
                flex: 0 0 45%; /* Approximately 45% width for the left panel */
                padding: 3rem 5rem;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: flex-start;
            }
            .right-panel {
                flex: 1; /* Takes remaining width */
                background: linear-gradient(135deg, #FFD1DC, #C9EEFF, #B9E6FF, #A7D9FF); /* Placeholder gradient */
                /* For a real image background, you would use: */
                /* background-image: url('URL_TO_YOUR_IMAGE'); */
                /* background-size: cover; */
                /* background-position: center; */
            }
            .stButton > button {
                width: 100%;
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                font-weight: bold;
                margin-top: 1rem;
            }
            .signup-button > button {
                background-color: #3B82F6; /* Blue */
                color: white;
                border: none;
            }
            .social-button {
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 0.5rem;
                border: 1px solid #D1D5DB; /* Gray border */
                border-radius: 0.5rem;
                cursor: pointer;
                transition: background-color 0.2s;
                height: 3rem; /* Fixed height for consistency */
            }
            .social-button:hover {
                background-color: #F3F4F6; /* Light gray on hover */
            }
            .or-divider {
                display: flex;
                align-items: center;
                text-align: center;
                margin: 1.5rem 0;
                color: #6B7280; /* Gray text */
            }
            .or-divider::before,
            .or-divider::after {
                content: '';
                flex: 1;
                border-bottom: 1px solid #E5E7EB; /* Light gray line */
            }
            .or-divider:not(:empty)::before {
                margin-right: .5em;
            }
            .or-divider:not(:empty)::after {
                margin-left: .5em;
            }
            .footer-text {
                position: absolute;
                bottom: 2rem;
                left: 5rem;
                color: #6B7280;
                font-size: 0.9rem;
            }
            .footer-text a {
                color: #3B82F6; /* Blue link */
                text-decoration: none;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([0.45, 0.55]) # Adjust column ratios

    with col_left:
        st.image("https://github.com/streamlit/docs/blob/main/docs/static/logo.png?raw=true", width=30) # Placeholder logo
        st.markdown("<h4 style='margin-top: 2rem; color: #6B7280;'>Start your journey</h4>", unsafe_allow_html=True)
        st.markdown("<h2>Sign Up to InsideBox</h2>", unsafe_allow_html=True)

        email = st.text_input("E-mail", value="example@email.com", key="email_input",
                              placeholder="example@email.com",
                              label_visibility="visible") # Added label visibility
        st.markdown(
            """
            <style>
                div[data-testid="stTextInput"] input {
                    padding-left: 2.5rem; /* Space for icon */
                }
                div[data-testid="stTextInput"] > div > div > label {
                    position: absolute;
                    top: -1.5rem; /* Adjust label position */
                    left: 0;
                    font-size: 0.9rem;
                    color: #4B5563;
                }
                div[data-testid="stTextInput"] > div:nth-child(2) {
                    position: relative;
                }
                div[data-testid="stTextInput"] > div:nth-child(2)::before {
                    content: '✉️'; /* Email icon */
                    position: absolute;
                    left: 0.75rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: #9CA3AF;
                    font-size: 1rem;
                }
            </style>
            """, unsafe_allow_html=True
        )


        password = st.text_input("Password", type="password", key="password_input",
                                 placeholder="••••••••",
                                 label_visibility="visible") # Added label visibility
        st.markdown(
            """
            <style>
                div[data-testid="stTextInput"]:nth-of-type(2) > div > div > label {
                    position: absolute;
                    top: -1.5rem; /* Adjust label position */
                    left: 0;
                    font-size: 0.9rem;
                    color: #4B5563;
                }
                div[data-testid="stTextInput"]:nth-of-type(2) > div:nth-child(2) {
                    position: relative;
                }
                div[data-testid="stTextInput"]:nth-of-type(2) > div:nth-child(2)::before {
                    content: '🔒'; /* Password icon */
                    position: absolute;
                    left: 0.75rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: #9CA3AF;
                    font-size: 1rem;
                }
            </style>
            """, unsafe_allow_html=True
        )


        if st.button("Sign Up", key="signup_button", help="Click to sign up"):
            st.success(f"Signing up with {email}...") # Placeholder action

        st.markdown("<div class='or-divider'>or sign up with</div>", unsafe_allow_html=True)

        col_social1, col_social2, col_social3 = st.columns(3)
        with col_social1:
            st.markdown("<div class='social-button'>🇫</div>", unsafe_allow_html=True) # Facebook placeholder
        with col_social2:
            st.markdown("<div class='social-button'>G</div>", unsafe_allow_html=True) # Google placeholder
        with col_social3:
            st.markdown("<div class='social-button'></div>", unsafe_allow_html=True) # Apple placeholder

        st.markdown(
            "<div class='footer-text'>Have an account? <a href='#'>Sign in</a></div>",
            unsafe_allow_html=True
        )

    with col_right:
        # This column will act as the background image container
        st.empty() # Placeholder for the visual background from the image


if __name__ == "__main__":
    login_page()
