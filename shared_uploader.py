import streamlit as st
import pandas as pd
import os

UPLOAD_DIR = "saved_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def upload_or_select_file(label, key=None):
    # اختيار ملف محفوظ
    saved_files = os.listdir(UPLOAD_DIR)
    selected_file = None
    if saved_files:
        choice = st.selectbox("📂 اختر ملف محفوظ", [""] + saved_files, key=f"{key}_select")
        if choice:
            selected_file = os.path.join(UPLOAD_DIR, choice)

    # رفع ملف جديد
    uploaded_file = st.file_uploader(label, key=key)
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_file = file_path

    return selected_file
