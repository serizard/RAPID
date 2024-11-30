import streamlit as st

def render_navbar():
    """Render the navigation bar."""
    nav_items = {
        "start": "Start",
        "instruction": "Instruction",
        "information": "User Information",
        "record": "Record Test",
        "result": "Result",
        "pdf": "Download Report"
    }
    
    nav_html = """
        <div class="navbar">
            {links}
        </div>
    """
    
    links = []
    for page, label in nav_items.items():
        active = "active" if st.session_state.page == page else ""
        links.append(f'<a href="/?page={page}" class="nav-item {active}">{label}</a>')
    
    st.markdown(nav_html.format(links="".join(links)), unsafe_allow_html=True)

def create_back_button(previous_page: str):
    """Create a back navigation button."""
    if st.button("← Back", key="back_button", type="secondary"):
        st.session_state.page = previous_page
        st.rerun()