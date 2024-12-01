import streamlit.components.v1 as components

_component_func = components.declare_component(
    "medical_report",
    url="http://localhost:3000"
)

def medical_report(patient_info, key=None):
    return _component_func(patient_info=patient_info, key=key)