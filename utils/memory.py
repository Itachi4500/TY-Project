import streamlit as st
import json
import time
from datetime import datetime
from collections import defaultdict
from io import StringIO

# --- INIT ---
def init_memory():
    if "memory" not in st.session_state:
        st.session_state.memory = {}
    if "memory_history" not in st.session_state:
        st.session_state.memory_history = []

# --- NESTED MEMORY HANDLER ---
def set_nested_key(data, key_path, value):
    keys = key_path.split('.')
    d = data
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value
    return data

def get_nested_key(data, key_path):
    keys = key_path.split('.')
    d = data
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d

# --- CORE FUNCTIONS ---
def remember(key: str, value):
    init_memory()
    st.session_state.memory = set_nested_key(st.session_state.memory, key, value)
    
    # Add timestamped history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.memory_history.append({
        "timestamp": timestamp,
        "key": key,
        "value": value
    })
    st.success(f"✅ Stored `{key}` at {timestamp}")

def recall(key: str):
    init_memory()
    return get_nested_key(st.session_state.memory, key)

def forget(key: str):
    init_memory()
    keys = key.split('.')
    d = st.session_state.memory
    try:
        for k in keys[:-1]:
            d = d[k]
        del d[keys[-1]]
        st.success(f"🗑️ Removed `{key}`")
    except (KeyError, TypeError):
        st.warning(f"⚠️ Key `{key}` not found.")

def clear_all_memory():
    st.session_state.memory = {}
    st.session_state.memory_history = []
    st.success("🧹 All memory and history cleared.")

# --- DISPLAY FUNCTIONS ---
def show_memory():
    init_memory()
    st.markdown("### 🧠 Current Memory")
    st.json(st.session_state.memory)

    if st.button("📥 Download Memory as JSON"):
        memory_str = json.dumps(st.session_state.memory, indent=2)
        st.download_button("⬇️ Download JSON", memory_str, file_name="memory.json", mime="application/json")

def show_memory_history():
    init_memory()
    if st.session_state.memory_history:
        st.markdown("### 🕒 Memory Change History")
        st.dataframe(pd.DataFrame(st.session_state.memory_history))
    else:
        st.info("ℹ️ No memory history recorded.")
