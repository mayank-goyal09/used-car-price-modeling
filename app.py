import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
import os
import tempfile
import gdown
import requests

# -----------------------
# Custom CSS - Car Theme
# -----------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Main Title Styling */
    h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00d4ff 0%, #ff0080 50%, #7928ca 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
        margin-bottom: 0 !important;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px #00d4ff); }
        to { filter: drop-shadow(0 0 20px #ff0080); }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(255, 0, 128, 0.3);
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a2e 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #00d4ff;
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
    }
    
    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border: 1px solid #00d4ff !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.5) !important;
        transform: scale(1.02);
    }
    
    /* Labels */
    label {
        color: #00d4ff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Predict Button */
    .stButton button {
        background: linear-gradient(135deg, #00d4ff 0%, #7928ca 100%) !important;
        color: white !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 3rem !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4) !important;
        transition: all 0.4s ease !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton button:hover {
        transform: scale(1.05) translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(255, 0, 128, 0.6) !important;
        background: linear-gradient(135deg, #ff0080 0%, #7928ca 100%) !important;
    }
    
    /* Success Message */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1) !important;
        border-left: 4px solid #00ff88 !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* DataFrame Styling */
    .dataframe {
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(121, 40, 202, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4);
    }
    
    .metric-title {
        color: #00d4ff;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Section Headers */
    h2, h3 {
        color: #00d4ff !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 2rem !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        margin: 2rem 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(0, 212, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #00d4ff !important;
        font-weight: 600 !important;
    }
    
    /* Info Box */
    .stInfo {
        background: rgba(0, 212, 255, 0.1) !important;
        border-left: 4px solid #00d4ff !important;
        border-radius: 10px !important;
    }
    
    /* Warning Box */
    .stWarning {
        background: rgba(255, 193, 7, 0.1) !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 10px !important;
    }
    
    /* Error Box */
    .stError {
        background: rgba(255, 0, 128, 0.1) !important;
        border-left: 4px solid #ff0080 !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------
# Load trained pipeline
# -----------------------
import os
import tempfile
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

def download_file_from_gdrive(file_id, output_path):
    # direct download URL format
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)
    response.raise_for_status()

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:  # handle Drive confirmation token
        response = session.get(
            url, params={"id": file_id, "confirm": token}, stream=True
        )
        response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_model():
    file_id = "1ZsYxc9lAA8uaUPmYIka3q0B97OfbPzkF"  # your Drive file ID
    model_name = "used_car_price_model.pkl"

    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, model_name)

    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading model (one-time)..."):
            download_file_from_gdrive(file_id, model_path)

    if os.path.getsize(model_path) < 10_000:
        raise RuntimeError("Downloaded file looks too small — likely HTML, not the model.")

    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error("❌ Model downloaded but failed to load.")
        st.exception(e)
        st.stop()

model = load_model()


# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="🏎️ Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load custom CSS
load_custom_css()


# -----------------------
# Header Section
# -----------------------
st.markdown("<h1>🏎️ USED CAR PRICE PREDICTOR 🚗</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>🔥 AI-Powered Price Estimation | Trained on Real CarDekho Data | Professional ML Model 🔥</p>", unsafe_allow_html=True)
st.markdown("---")


# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.markdown("## 🚘 VEHICLE SPECIFICATIONS")
st.sidebar.markdown("---")


# Basic Info Section
with st.sidebar.expander("📍 LOCATION & BASIC INFO", expanded=True):
    loc = st.text_input("🗺️ Location", "mumbai g.p.o.", key="loc")
    city = st.text_input("🏙️ City", "mumbai", key="city")
    myear = st.number_input("📅 Model Year", min_value=1990, max_value=2025, value=2018, step=1)
    km = st.number_input("📏 Kilometers Driven", min_value=0, max_value=500000, value=35000, step=1000)


# Vehicle Type Section
with st.sidebar.expander("🚙 VEHICLE TYPE & FUEL", expanded=True):
    body = st.selectbox("🏗️ Body Type", ["hatchback", "sedan", "suv"])
    fuel = st.selectbox("⛽ Fuel Type", ["petrol", "diesel", "cng", "lpg", "electric"])
    transmission = st.selectbox("⚙️ Transmission", ["manual", "automatic"])
    owner_type = st.selectbox("👤 Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])


# Performance Section
with st.sidebar.expander("⚡ ENGINE & PERFORMANCE", expanded=True):
    max_power = st.number_input("💪 Max Power (bhp)", min_value=20.0, max_value=600.0, value=82.0, step=1.0)
    seats = st.number_input("🪑 Seats", min_value=2, max_value=10, value=5, step=1)


# Dimensions Section
with st.sidebar.expander("📐 DIMENSIONS & SPECS", expanded=False):
    length = st.number_input("📏 Length (mm)", min_value=2500.0, max_value=6000.0, value=3850.0, step=10.0)
    width = st.number_input("↔️ Width (mm)", min_value=1400.0, max_value=2500.0, value=1690.0, step=10.0)
    height = st.number_input("⬆️ Height (mm)", min_value=1200.0, max_value=2500.0, value=1530.0, step=10.0)
    wheel_base = st.number_input("🛞 Wheel Base (mm)", min_value=2000.0, max_value=4000.0, value=2450.0, step=10.0)


# Additional Info
with st.sidebar.expander("📸 ADDITIONAL INFO", expanded=False):
    imgCount = st.number_input("📷 Number of Images", min_value=0, max_value=50, value=8)


st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro Tip:** More images = higher listing visibility!")


# -----------------------
# Main Content Area
# -----------------------

# Create three columns for key metrics display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🚗 Body Type</div>
        <div class="metric-value">{body.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">⛽ Fuel</div>
        <div class="metric-value">{fuel.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📅 Year</div>
        <div class="metric-value">{myear}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📏 KMs</div>
        <div class="metric-value">{km:,}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# Build input DataFrame
input_data = {
    "loc": [loc],
    "City": [city],
    "myear": [myear],
    "km": [float(km)],
    "body": [body],
    "fuel": [fuel],
    "transmission": [transmission],
    "owner_type": [owner_type],
    "imgCount": [int(imgCount)],
    "Max Power Delivered": [float(max_power)],
    "Seats": [int(seats)],
    "Length": [float(length)],
    "Width": [float(width)],
    "Height": [float(height)],
    "Wheel Base": [float(wheel_base)],
}

input_df = pd.DataFrame(input_data)


# Align with training columns
TRAIN_COLUMNS = None

if TRAIN_COLUMNS is not None:
    for col in TRAIN_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df = input_df[TRAIN_COLUMNS]


# Input Summary in Expander
with st.expander("🔍 VIEW COMPLETE INPUT SPECIFICATIONS", expanded=False):
    st.dataframe(input_df, use_container_width=True)


# -----------------------
# Predict Button
# -----------------------
st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("🔮 PREDICT PRICE", use_container_width=True):
        with st.spinner("🚀 AI Engine Analyzing Your Car..."):
            try:
                pred = model.predict(input_df)[0]
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display result in a beautiful card
                st.success(f"### 💰 Estimated Listing Price: ₹ {pred:,.0f}")
                
                st.balloons()
                
                # Additional insights
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_insight1, col_insight2, col_insight3 = st.columns(3)
                
                with col_insight1:
                    st.info(f"**📊 Price Range:** ₹ {pred*0.92:,.0f} - ₹ {pred*1.08:,.0f}")
                
                with col_insight2:
                    depreciation = (2025 - myear) * 0.12
                    st.warning(f"**📉 Age Impact:** ~{depreciation*100:.0f}% depreciation")
                
                with col_insight3:
                    km_condition = "Excellent" if km < 20000 else "Good" if km < 50000 else "Fair"
                    st.info(f"**🛞 Mileage Status:** {km_condition}")
                
            except Exception as e:
                st.error(f"❌ **Prediction Failed:** {e}")
                st.info("🔧 **Troubleshooting:** Ensure your input columns match the training features.")


# -----------------------
# Footer Section
# -----------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("### 🎯 About")
    st.markdown("Built with **Random Forest ML** trained on real-world CarDekho data")

with footer_col2:
    st.markdown("### 📊 Accuracy")
    st.markdown("Model achieves **high prediction accuracy** on test data")

with footer_col3:
    st.markdown("### 🚀 Tech Stack")
    st.markdown("**Python** • **Streamlit** • **Scikit-Learn** • **Pandas**")


st.markdown(f"""
<hr style="margin-top:2.2rem; background: linear-gradient(90deg, transparent, #00d4ff, transparent); 
           border:none; height:2px;">
<div style="text-align:center; padding:1.5rem 0; font-size:0.9rem; 
           background: rgba(0,212,255,0.05); border-radius:15px; margin-top:1rem;">
  <div style="color:#00d4ff; margin-bottom:0.5rem;">
    © 2025 <span style="font-weight:900; font-size:1.1rem; 
    background: linear-gradient(90deg, #00d4ff, #ff0080, #7928ca); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    AutoPredict AI</span>
  </div>
  <div style="color:#a0a0a0; font-size:0.85rem; margin-bottom:1rem;">
    🚗 Used Car Price Intelligence · Powered by Machine Learning
  </div>
  <div style="margin-bottom:0.8rem;">
    <span style="color:#ffffff; font-weight:600;">Crafted by </span>
    <span style="font-weight:900; color:#ff0080; font-size:1.05rem;">Mayank Goyal</span>
  </div>
  <div style="margin-bottom:1rem;">
    <a href="https://www.linkedin.com/in/mayank-goyal-4b8756363" target="_blank"
       style="display:inline-block; background:rgba(0,212,255,0.1); color:#00d4ff; 
       text-decoration:none; font-weight:700; padding:0.5rem 1.2rem; 
       border-radius:25px; margin:0.3rem; border:1px solid rgba(0,212,255,0.3);
       transition: all 0.3s ease;">
       💼 LinkedIn</a>
    <a href="https://github.com/mayank-goyal09" target="_blank"
       style="display:inline-block; background:rgba(255,0,128,0.1); color:#ff0080; 
       text-decoration:none; font-weight:700; padding:0.5rem 1.2rem; 
       border-radius:25px; margin:0.3rem; border:1px solid rgba(255,0,128,0.3);
       transition: all 0.3s ease;">
       💻 GitHub</a>
  </div>
  <div style="font-size:0.75rem; color:#666; letter-spacing:1px;">
    ⚡ Random Forest Algorithm · 📊 CarDekho Real Data · 🔮 Real-time Price Estimation
  </div>
</div>

<style>
a:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(0,212,255,0.4) !important;
}
</style>
""", unsafe_allow_html=True)



