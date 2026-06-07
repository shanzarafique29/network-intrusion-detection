import streamlit as st
import pickle
import numpy as np
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Network Anomaly Detector", page_icon="🛡️", layout="centered")

# --- 2. PATHS TO BACKEND ASSETS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, '..', 'backend', 'best_rf_model.pkl')
CNN_MODEL_PATH = os.path.join(BASE_DIR, '..', 'backend', 'best_cnn_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'backend', 'scaler.pkl')

# --- 3. LOAD ASSETS WITH VERSION BYPASS ---
@st.cache_resource
def load_assets():
    # Load Random Forest (Yeh hamesha safely load hoga)
    with open(RF_MODEL_PATH, 'rb') as m_file:
        rf_model = pickle.load(m_file)
    
    # Load Scaler
    with open(SCALER_PATH, 'rb') as s_file:
        scaler = pickle.load(s_file)
    
    # Try loading CNN safely
    cnn_model = None
    try:
        import tensorflow as tf
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception as e:
        # If version mismatch happens locally, it won't crash the app
        pass
        
    return rf_model, cnn_model, scaler

rf_model, cnn_model, scaler = load_assets()

# Dynamic Sidebar Configuration based on what loaded successfully
if cnn_model is not None:
    st.sidebar.success("✅ ML (Random Forest) & DL (CNN) Models Loaded!")
    model_options = ["Random Forest (ML)", "CNN (Deep Learning)"]
else:
    st.sidebar.warning("⚠️ Running in ML Mode (CNN version mismatch locally)")
    model_options = ["Random Forest (ML)"]

# --- 4. APP USER INTERFACE (UI) ---
st.title("🛡️ Hybrid Network Intrusion & Anomaly Detection System")
st.write("Toggle between Machine Learning (Random Forest) and Deep Learning (CNN) to detect malicious traffic.")
st.markdown("---")

selected_model = st.sidebar.selectbox("🤖 Choose Architecture / Model:", model_options)

st.subheader("📊 Traffic Features Input")
col1, col2 = st.columns(2)

with col1:
    packet_length = st.number_input("Packet Length (bytes):", min_value=0, value=1000)
    duration = st.number_input("Duration (seconds):", min_value=0.0, value=1.5)
    source_port = st.number_input("Source Port:", min_value=0, max_value=65535, value=443)

with col2:
    dest_port = st.number_input("Destination Port:", min_value=0, max_value=65535, value=80)
    header_length = st.number_input("Header Length:", min_value=0, value=40)
    protocol = st.selectbox("Protocol Type:", options=[0, 1, 2], format_func=lambda x: ['TCP', 'UDP', 'ICMP'][x])

expected_features = rf_model.n_features_in_
user_inputs = [packet_length, duration, source_port, dest_port, header_length, protocol]

while len(user_inputs) < expected_features:
    user_inputs.append(0)

# --- 5. PREDICTION LOGIC ---
if st.button("🚀 Analyze Traffic", type="primary"):
    features_array = np.array([user_inputs])
    scaled_inputs = scaler.transform(features_array)
    
    prediction = 0
    confidence = 0.0
    
    if selected_model == "Random Forest (ML)":
        prediction = rf_model.predict(scaled_inputs)[0]
        confidence = rf_model.predict_proba(scaled_inputs)[0][prediction] * 100
    
    elif selected_model == "CNN (Deep Learning)" and cnn_model is not None:
        import tensorflow as tf
        cnn_inputs = scaled_inputs.reshape((scaled_inputs.shape[0], scaled_inputs.shape[1], 1))
        prob = cnn_model.predict(cnn_inputs).flatten()[0]
        
        if prob > 0.5:
            prediction = 1
            confidence = prob * 100
        else:
            prediction = 0
            confidence = (1 - prob) * 100

    st.markdown("---")
    st.subheader(f"🎯 Result Analysis ({selected_model}):")
    
    if prediction == 1:
        st.error(f"🚨 **WARNING: Malicious Traffic / Anomaly Detected!** (Confidence: {confidence:.2f}%)")
    else:
        st.success(f"💚 **SAFE: Normal Network Traffic.** (Confidence: {confidence:.2f}%)")