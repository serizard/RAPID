import streamlit as st
from pathlib import Path

def show_spinner():
    """Show loading spinner."""
    st.markdown("""
        <div class="spinner-container">
            <div class="spinner"></div>
        </div>
    """, unsafe_allow_html=True)

def load_css(css_path: Path):
    """Load CSS file."""
    try:
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        print(f"CSS loading error: {str(e)}")
