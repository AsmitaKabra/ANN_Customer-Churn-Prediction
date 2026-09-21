"""Customer Churn Prediction - Streamlit app.

Serves the trained ANN (churn_model.keras) using the exact preprocessing
pipeline from prediction.ipynb:
    1. Label-encode Gender
    2. One-hot encode Geography and append the columns after the numeric features
    3. Standard-scale the 12 features
    4. Predict and apply a 0.5 decision threshold
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.keras"
GEO_ENCODER_PATH = BASE_DIR / "onehot_encoder_geo.pkl"
GENDER_ENCODER_PATH = BASE_DIR / "label_encoder_gender.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

THRESHOLD = 0.5

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #0E1B2C;
    --body: #34455A;
    --muted: #6B7A8C;
    --paper: #F3F5F8;
    --card: #FFFFFF;
    --line: #DDE3EA;
    --accent: #2447C7;
    --accent-dark: #1B3AA6;
    --stay: #0F7B67;
    --leave: #C93B22;
}

/* ---- Page ---- */
.stApp, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.stApp, .stApp button, .stApp input, .stApp label, .stApp p, .stApp li {
    font-family: 'Manrope', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
.stApp .block-container { max-width: 860px; padding-top: 2.5rem; padding-bottom: 3rem; }

/* ---- Hero ---- */
.stApp .hero {
    background: var(--ink) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='520' height='180' viewBox='0 0 520 180' fill='none'%3E%3Cpath d='M0 28C110 30 170 46 230 96S380 150 520 146' stroke='white' stroke-opacity='.16' stroke-width='2' stroke-linecap='round'/%3E%3Ccircle cx='230' cy='96' r='5' fill='white' fill-opacity='.35'/%3E%3C/svg%3E") no-repeat right bottom / 60% auto;
    border-radius: 18px;
    padding: 2rem 2.25rem 2.1rem;
    margin-bottom: 1.25rem;
}
.stApp .hero-tag {
    display: inline-block; padding: .2rem .75rem; border-radius: 999px;
    border: 1px solid rgba(255,255,255,.25); color: #B4C1D3;
    font-size: .78rem; font-weight: 600;
}
.stApp .hero-title {
    color: #fff; font-size: 2.15rem; font-weight: 800; line-height: 1.15;
    letter-spacing: -0.025em; margin: 1rem 0 .5rem 0;
}
.stApp .hero-sub { color: #B4C1D3; font-size: 1.02rem; line-height: 1.5; max-width: 34rem; }

/* ---- Form card ---- */
.stApp [data-testid="stForm"] {
    background: var(--card); border: 1px solid var(--line); border-radius: 18px;
    padding: 1.6rem 1.75rem 1.5rem;
    box-shadow: 0 1px 2px rgba(14,27,44,.04), 0 8px 24px rgba(14,27,44,.04);
}
.stApp .section-title {
    color: var(--ink); font-size: .98rem; font-weight: 700;
    margin: .4rem 0 0 0; padding-top: 1.1rem; border-top: 1px solid var(--line);
}
.stApp .section-title.first { margin: 0; padding-top: 0; border-top: 0; }

/* ---- Inputs ---- */
.stApp [data-testid="stWidgetLabel"] p { color: var(--body); font-size: .84rem; font-weight: 600; }
.stApp div[data-baseweb="input"],
.stApp div[data-baseweb="select"] > div {
    background: #fff; border: 1px solid #CBD5E1; border-radius: 10px; color: var(--ink);
}
.stApp div[data-baseweb="base-input"] { background: transparent; }
.stApp div[data-baseweb="input"] input,
.stApp div[data-baseweb="select"] * { color: var(--ink); -webkit-text-fill-color: var(--ink); }
.stApp div[data-baseweb="input"]:focus-within,
.stApp div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent); box-shadow: 0 0 0 3px rgba(36,71,199,.16);
}
.stApp [data-testid="stNumberInput"] button { background: #F1F4F8; color: var(--ink); border: 0; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { background: #fff; color: var(--ink); }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { background: #EEF2F8; }

/* ---- Predict button ---- */
.stApp [data-testid="stFormSubmitButton"] { width: 100%; }
.stApp [data-testid="stFormSubmitButton"] button,
.stApp button[data-testid="stBaseButton-primaryFormSubmit"],
.stApp button[kind="primaryFormSubmit"] {
    width: 100%; min-height: 3rem; background: var(--accent); color: #fff;
    border: 0; border-radius: 10px; font-size: 1rem; font-weight: 700;
    transition: background .15s ease;
}
.stApp [data-testid="stFormSubmitButton"] button p,
.stApp button[data-testid="stBaseButton-primaryFormSubmit"] p,
.stApp button[kind="primaryFormSubmit"] p { color: #fff; font-weight: 700; }
.stApp [data-testid="stFormSubmitButton"] button:hover,
.stApp button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
.stApp button[kind="primaryFormSubmit"]:hover { background: var(--accent-dark); }
.stApp [data-testid="stFormSubmitButton"] button:focus-visible {
    outline: 3px solid rgba(36,71,199,.35); outline-offset: 2px;
}

/* ---- Result ---- */
.stApp .result {
    border: 1px solid; border-radius: 18px; padding: 1.5rem 1.75rem 1.6rem;
    margin-top: 1.25rem; animation: rise .4s ease-out;
}
.stApp .result.low  { background: #EAF6F2; border-color: #BFE2D8; --tone: var(--stay); }
.stApp .result.high { background: #FCEEEA; border-color: #F2C8BD; --tone: var(--leave); }
.stApp .result-head { display: flex; align-items: center; gap: .8rem; }
.stApp .badge {
    width: 2.25rem; height: 2.25rem; border-radius: 50%; background: var(--tone); color: #fff;
    display: grid; place-items: center; font-size: 1.1rem; font-weight: 800;
}
.stApp .verdict { color: var(--tone); font-size: 1.4rem; font-weight: 800; letter-spacing: -0.01em; }
.stApp .prob { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; margin: 1.1rem 0 1.6rem; }
.stApp .prob-value {
    color: var(--ink); font-size: 3.2rem; font-weight: 800; line-height: 1;
    letter-spacing: -0.03em; font-variant-numeric: tabular-nums;
}
.stApp .prob-label { color: var(--body); font-size: 1rem; font-weight: 500; }

.stApp .track {
    position: relative; height: 12px; border-radius: 999px;
    background: linear-gradient(90deg, rgba(14,27,44,.09) 0 50%, rgba(201,59,34,.18) 50% 100%);
}
.stApp .fill {
    position: relative; height: 100%; border-radius: 999px; background: var(--tone);
    animation: grow .8s cubic-bezier(.2,.7,.2,1);
}
.stApp .fill::after {
    content: ""; position: absolute; right: -3px; top: 50%; width: 20px; height: 20px;
    transform: translateY(-50%); box-sizing: border-box; border-radius: 50%;
    background: var(--tone); border: 3px solid #fff; box-shadow: 0 1px 3px rgba(14,27,44,.3);
}
.stApp .tick {
    position: absolute; left: 50%; top: -5px; width: 2px; height: 22px; margin-left: -1px;
    border-radius: 1px; background: var(--ink); opacity: .55;
}
.stApp .scale {
    position: relative; height: 1.1rem; margin-top: .8rem;
    color: var(--body); font-size: .78rem; font-weight: 600;
}
.stApp .scale span { position: absolute; top: 0; }
.stApp .scale .s0 { left: 0; }
.stApp .scale .s50 { left: 50%; transform: translateX(-50%); }
.stApp .scale .s100 { right: 0; }
.stApp .explain { margin-top: 1.1rem; color: var(--body); font-size: .92rem; line-height: 1.5; }

.stApp .footer { margin-top: 2rem; text-align: center; color: var(--muted); font-size: .82rem; }

@keyframes rise { from { opacity: 0; transform: translateY(8px); } }
@keyframes grow { from { width: 0; } }
@media (prefers-reduced-motion: reduce) {
    .stApp .result, .stApp .fill { animation: none; }
}
@media (max-width: 640px) {
    .stApp .hero { padding: 1.5rem; }
    .stApp .hero-title { font-size: 1.7rem; }
    .stApp .prob-value { font-size: 2.6rem; }
}
</style>
"""

HERO_HTML = (
    '<div class="hero">'
    '<span class="hero-tag">Artificial neural network</span>'
    '<div class="hero-title">Customer churn prediction</div>'
    '<div class="hero-sub">Enter a bank customer\'s details to estimate '
    "how likely they are to leave.</div>"
    "</div>"
)

FOOTER_HTML = (
    '<div class="footer">Built with Streamlit, TensorFlow and scikit-learn</div>'
)

st.markdown(STYLES, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model and preprocessors...")
def load_artifacts():
    """Load the Keras model and the fitted preprocessors once per session."""
    required = [MODEL_PATH, GEO_ENCODER_PATH, GENDER_ENCODER_PATH, SCALER_PATH]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(missing)
            + f". Place them in the same folder as app.py ({BASE_DIR})."
        )

    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)

    with open(GEO_ENCODER_PATH, "rb") as f:
        onehot_encoder_geo = pickle.load(f)
    with open(GENDER_ENCODER_PATH, "rb") as f:
        label_encoder_gender = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, onehot_encoder_geo, label_encoder_gender, scaler


# --------------------------------------------------------------------------- #
# Preprocessing + prediction (mirrors prediction.ipynb)
# --------------------------------------------------------------------------- #
def predict_churn(input_data, model, onehot_encoder_geo, label_encoder_gender, scaler):
    """Return the churn probability (0-1) for one customer."""
    input_df = pd.DataFrame([input_data])

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform(
        pd.DataFrame([[input_data["Geography"]]], columns=["Geography"])
    ).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Label-encode Gender
    input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])

    # Feature order: numeric/binary columns, then Geography_* columns
    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled, verbose=0)
    return float(prediction[0][0])


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def result_html(probability):
    """Build the result card: verdict, probability and a threshold gauge."""
    likely = probability > THRESHOLD
    pct = probability * 100
    fill = min(max(pct, 0.6), 100)
    threshold_pct = f"{THRESHOLD:.0%}"

    if likely:
        tone, badge, verdict = "high", "!", "Likely to Churn"
        explain = (
            f"The predicted probability is above the {threshold_pct} decision "
            "threshold, so this customer is classified as likely to leave."
        )
    else:
        tone, badge, verdict = "low", "&#10003;", "Unlikely to Churn"
        explain = (
            f"The predicted probability is at or below the {threshold_pct} decision "
            "threshold, so this customer is classified as unlikely to leave."
        )

    return (
        f'<div class="result {tone}" role="status">'
        f'<div class="result-head"><div class="badge">{badge}</div>'
        f'<div class="verdict">{verdict}</div></div>'
        f'<div class="prob"><span class="prob-value">{pct:.2f}%</span>'
        '<span class="prob-label">churn probability</span></div>'
        f'<div class="track"><div class="fill" style="width: {fill:.2f}%"></div>'
        '<div class="tick"></div></div>'
        '<div class="scale"><span class="s0">0%</span>'
        f'<span class="s50">{threshold_pct} threshold</span>'
        '<span class="s100">100%</span></div>'
        f'<div class="explain">{explain}</div>'
        "</div>"
    )


def section(title, first=False):
    css_class = "section-title first" if first else "section-title"
    st.markdown(f'<div class="{css_class}">{title}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.markdown(HERO_HTML, unsafe_allow_html=True)

try:
    model, onehot_encoder_geo, label_encoder_gender, scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(
        "Could not load the model or preprocessors. Check that the installed "
        "TensorFlow / scikit-learn versions match requirements.txt."
    )
    st.exception(e)
    st.stop()

geography_options = list(onehot_encoder_geo.categories_[0])
gender_options = list(label_encoder_gender.classes_)

with st.form("churn_form"):
    section("Customer profile", first=True)
    c1, c2, c3, c4 = st.columns(4)
    geography = c1.selectbox("Geography", geography_options)
    gender = c2.selectbox("Gender", gender_options)
    age = c3.number_input("Age", min_value=18, max_value=92, value=40, step=1)
    tenure = c4.number_input("Tenure (years)", min_value=0, max_value=10, value=3, step=1)

    section("Financials")
    c1, c2, c3 = st.columns(3)
    credit_score = c1.number_input("Credit Score", min_value=300, max_value=850, value=600, step=1)
    balance = c2.number_input("Balance", min_value=0, value=60000, step=1000, format="%d")
    estimated_salary = c3.number_input(
        "Estimated Salary", min_value=0, value=50000, step=1000, format="%d"
    )

    section("Products and activity")
    c1, c2, c3 = st.columns(3)
    num_of_products = c1.number_input("Number of Products", min_value=1, max_value=4, value=2, step=1)
    has_cr_card = c2.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = c3.selectbox("Is Active Member", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Churn", type="primary")

if submitted:
    # Same keys and order as input_data in prediction.ipynb
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
    }

    try:
        with st.spinner("Predicting..."):
            probability = predict_churn(
                input_data, model, onehot_encoder_geo, label_encoder_gender, scaler
            )
        st.markdown(result_html(probability), unsafe_allow_html=True)
    except Exception as e:
        st.error("Prediction failed. Please check the inputs and try again.")
        st.exception(e)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
