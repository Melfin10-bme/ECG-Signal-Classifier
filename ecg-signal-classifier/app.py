import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocess import preprocess_ecg
from utils.features import extract_time_domain_features, classify_ecg
from utils.plotter import plot_ecg
import plotly.graph_objs as go

st.set_page_config(page_title="ECG Signal Classifier", layout="wide")

st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to:", ["ECG Classifier", "About"])

if page == "ECG Classifier":
    st.title("🫀 ECG Signal Classifier")

    uploaded_file = st.file_uploader("Upload ECG CSV file", type=["csv"])
    sample_choice = st.selectbox("Or use sample ECG", ["None", "Normal ECG", "Arrhythmia ECG"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif sample_choice != "None":
        if sample_choice == "Normal ECG":
            df = pd.read_csv("sample_data/normal_ecg.csv")
        else:
            df = pd.read_csv("sample_data/arrhythmia_ecg.csv")
    else:
        df = None

    if df is not None:
        st.write("### Raw ECG Data Preview")
        st.write(df.head())

        # Column selection
        col_time = st.selectbox("Select Time Column", df.columns, index=0)
        col_ecg = st.selectbox("Select ECG Column", df.columns, index=1)

        time = df[col_time].values
        ecg = df[col_ecg].values

        # Preprocess
        fs = 250  # assume 250Hz if unknown
        ecg_filtered = preprocess_ecg(ecg, fs)

        # Plot
        fig = plot_ecg(time[:1000], ecg_filtered[:1000])
        st.plotly_chart(fig, use_container_width=True)

        # Features
        features = extract_time_domain_features(ecg_filtered, fs)
        st.subheader("📊 Extracted Features")
        st.json(features)

        # Classification
        result, confidence = classify_ecg(features)
        st.subheader("🩺 Classification Result")
        st.success(f"{result} (Confidence: {confidence:.1f}%)")

elif page == "About":
    st.title("ℹ️ About This App")
    st.write("""
    This app demonstrates an ECG signal classifier built in **Python + Streamlit**.  
    - Upload an ECG file or use provided samples  
    - Signal is filtered & preprocessed  
    - HRV features are extracted  
    - Classified as **Normal** or **Arrhythmia**  

    Educational project for biomedical IoT/ML applications.
    """)
