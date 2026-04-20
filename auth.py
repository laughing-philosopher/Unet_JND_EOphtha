"""
auth.py — Local authentication module for OphthApp.
- Stores users in users.json (hashed passwords, no plaintext).
- Default admin: admin / password (created on first run).
- Doctors can self-register.
- Session state is used to track login across Streamlit reruns.
"""

import json
import os
import hashlib
import streamlit as st
import sys

from helper import get_base64_image, get_path
from translations import get_text  # Added import for translations

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def t(key):
    """Global translation helper for auth."""
    return get_text(st.session_state.get("lang_code", "en"), key)


def _hash(password: str) -> str:
    """SHA-256 hash of a password string."""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict:
    """Load users dict from disk. Auto-creates file with default admin if missing."""
    if not os.path.exists(USERS_FILE):
        default = {
            "admin": {
                "password_hash": _hash("password"),
                "role": "admin",
                "full_name": "Administrator",
            }
        }
        _save_users(default)
        return default

    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    """Persist users dict to disk."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def login(username: str, password: str) -> bool:
    """Return True and set session state if credentials are valid."""
    users = _load_users()
    username = username.strip().lower()
    if username in users and users[username]["password_hash"] == _hash(password):
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.session_state["full_name"] = users[username].get("full_name", username)
        st.session_state["role"] = users[username].get("role", "doctor")
        return True
    return False


def logout() -> None:
    """Clear login session state."""
    for key in ["logged_in", "username", "full_name", "role"]:
        st.session_state.pop(key, None)


def register(username: str, password: str, full_name: str) -> tuple[bool, str]:
    """
    Register a new doctor account.
    Returns (success: bool, message: str).
    """
    users = _load_users()
    username = username.strip().lower()

    if not username or not password or not full_name.strip():
        # Passed back to UI, can remain English or be translated at the UI level
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if username in users:
        return False, "Username already exists. Please choose another."

    users[username] = {
        "password_hash": _hash(password),
        "role": "doctor",
        "full_name": full_name.strip(),
    }
    _save_users(users)
    return True, "Account created successfully! You can now log in."


def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def current_user() -> dict:
    """Return dict with username, full_name, role for the logged-in user."""
    return {
        "username": st.session_state.get("username", ""),
        "full_name": st.session_state.get("full_name", ""),
        "role": st.session_state.get("role", ""),
    }


# ---------------------------------------------------------------------------
# UI — Login / Register screen (call this from app.py)
# ---------------------------------------------------------------------------

def show_auth_screen() -> None:
    """
    Renders the full login / register UI.
    Call this at the top of main() before any other content.
    Blocks rendering of the rest of the app until logged in.
    """
    # Force language code to exist before rendering the UI to prevent errors
    if "lang_code" not in st.session_state:
        st.session_state["lang_code"] = "en"

    st.set_page_config(layout="centered", page_title=f"Aakhi — {t('login_btn')}")

    # Center the card with columns
    _, center, _ = st.columns([1, 2, 1])
    logo_path = get_path("aakhi_logo.png")    
    img_base64 = get_base64_image(logo_path)

    with center:
        st.markdown(
            f"""
            <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
                <div style='display: inline-flex; align-items: center; justify-content: center;'>
                    <img src='data:image/png;base64,{img_base64}' style='width: 80px; margin-right: 15px;'>
                    <h2 style='margin: 0;'>AAKHI</h2>
                </div>
                <p style='color:gray; margin-top:4px;'>{t('retina_analysis')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Use a flag to switch between Login and Register views
        if "auth_page" not in st.session_state:
            st.session_state["auth_page"] = "login"

        # ---- LOGIN VIEW ----
        if st.session_state["auth_page"] == "login":
            if st.session_state.pop("_registered_ok", False):
                st.success(t("account_created_success"))

            st.markdown(f"#### {t('sign_in_prompt')}")
            login_user = st.text_input(t("username_label"), key="login_username", placeholder="Enter username")
            login_pass = st.text_input(t("password_label"), type="password", key="login_password", placeholder="Enter password")

            if st.button(t("login_btn"), use_container_width=True, key="login_btn"):
                if not login_user or not login_pass:
                    st.error(t("enter_user_pass_error"))
                elif login(login_user, login_pass):
                    st.rerun()
                else:
                    st.error(t("invalid_credentials_error"))

            st.markdown("---")
            st.markdown(t("no_account_prompt"))
            if st.button(t("create_account_btn"), use_container_width=True, key="go_register"):
                st.session_state["auth_page"] = "register"
                st.rerun()

        # ---- REGISTER VIEW ----
        else:
            st.markdown(f"#### {t('create_doctor_account')}")
            reg_name = st.text_input(t("full_name_label"), key="reg_fullname", placeholder="Dr. Jane Smith")
            reg_user = st.text_input(t("username_label"), key="reg_username", placeholder="Choose a username (min 3 chars)")
            reg_pass = st.text_input(t("password_label"), type="password", key="reg_password", placeholder="Choose a password (min 6 chars)")
            reg_pass2 = st.text_input(t("confirm_password_label"), type="password", key="reg_password2", placeholder="Repeat password")

            if st.button(t("create_account_submit_btn"), use_container_width=True, key="register_btn"):
                if reg_pass != reg_pass2:
                    st.error(t("passwords_mismatch_error"))
                else:
                    ok, msg = register(reg_user, reg_pass, reg_name)
                    if ok:
                        st.session_state["auth_page"] = "login"
                        st.session_state["_registered_ok"] = True
                        st.rerun()
                    else:
                        st.error(msg)  # Keeps backend validation errors standard

            st.markdown("---")
            st.markdown(t("already_have_account_prompt"))
            if st.button(t("back_to_login_btn"), use_container_width=True, key="go_login"):
                st.session_state["auth_page"] = "login"
                st.rerun()