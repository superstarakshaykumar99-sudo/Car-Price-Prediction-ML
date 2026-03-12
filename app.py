import streamlit as st
import pickle
import os
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

.block-container {
    padding-top: 2rem;
    max-width: 780px;
}

.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa, #00a8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.hero-sub {
    text-align: center;
    color: #8e9ab5;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}

.card-title {
    color: #00d4aa;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.result-box {
    background: linear-gradient(135deg, rgba(0,212,170,0.15), rgba(0,168,255,0.15));
    border: 1.5px solid #00d4aa;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-label {
    color: #8e9ab5;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.result-price {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa, #00a8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.error-box {
    background: rgba(255,80,80,0.1);
    border: 1px solid rgba(255,80,80,0.4);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #ff6b6b;
    margin-top: 1rem;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d4aa, #00a8ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,170,0.35) !important;
}

label, .stSelectbox label, .stNumberInput label {
    color: #c8d0e0 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

.stNumberInput input, .stTextInput input {
    border-radius: 10px !important;
}

.stSelectbox > div > div {
    border-radius: 10px !important;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(BASE_DIR, "saved_models", "RandomForestRegressor.pkl")
    scaler_path = os.path.join(BASE_DIR, "saved_scaling", "scaler.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()


def format_price(value):
    if value >= 10_000_000:
        return f"₹ {value/10_000_000:.2f} Cr"
    elif value >= 100_000:
        return f"₹ {value/100_000:.2f} Lakhs"
    else:
        return f"₹ {value:,.0f}"


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🚗 Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter car details below to get an instant AI-powered price estimate</div>', unsafe_allow_html=True)

# ── Card 1: Basic Info ────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📋 Basic Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    car_name    = st.text_input("Car Name", placeholder="e.g. Maruti Swift")
    km_driven   = st.number_input("KM Driven", min_value=0, max_value=1_000_000, value=30000, step=1000)
    engine_cc   = st.number_input("Engine (CC)", min_value=500, max_value=6000, value=1200, step=100)
with col2:
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=3, step=1)
    mileage     = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=100.0, value=18.0, step=0.5)
    max_power   = st.number_input("Max Power (bhp)", min_value=0.0, max_value=1000.0, value=85.0, step=1.0)

seats = st.select_slider("Number of Seats", options=[2, 4, 5, 6, 7], value=5)
st.markdown('</div>', unsafe_allow_html=True)

# ── Card 2: Car Type ─────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">⚙️ Car Type</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
with col2:
    fuel_type   = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
with col3:
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Price"):
    try:
        input_values = []

        # Numeric
        input_values.append(int(vehicle_age))
        input_values.append(int(km_driven))
        input_values.append(float(mileage))
        input_values.append(int(engine_cc))
        input_values.append(float(max_power))
        input_values.append(int(seats))

        # Seller Type (3)
        seller_enc = {"Dealer": [1,0,0], "Individual": [0,1,0], "Trustmark Dealer": [0,0,1]}
        input_values.extend(seller_enc[seller_type])

        # Fuel Type (5)
        fuel_enc = {"CNG": [1,0,0,0,0], "Diesel": [0,1,0,0,0], "Electric": [0,0,1,0,0],
                    "LPG": [0,0,0,1,0], "Petrol": [0,0,0,0,1]}
        input_values.extend(fuel_enc[fuel_type])

        # Transmission (2)
        trans_enc = {"Automatic": [1,0], "Manual": [0,1]}
        input_values.extend(trans_enc[transmission])

        if len(input_values) != 16:
            st.markdown(f'<div class="error-box">⚠️ Input size mismatch: expected 16, got {len(input_values)}</div>', unsafe_allow_html=True)
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled = scaler.transform([input_values])
                prediction = model.predict(scaled)[0]

            price_str = format_price(prediction)
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Estimated Market Price</div>
                <div class="result-price">{price_str}</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f'<div class="error-box">⚠️ Error: {e}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8e9ab5; font-size:0.8rem;'>"
    "Built with ❤️ using Streamlit & Random Forest | "
    "<a href='https://github.com/superstarakshaykumar99-sudo/Car-Price-Prediction-ML' "
    "style='color:#00d4aa;'>GitHub</a></p>",
    unsafe_allow_html=True
)
