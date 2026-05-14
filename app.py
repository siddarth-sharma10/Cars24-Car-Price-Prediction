import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Cars24 Price Predictor",
    page_icon=":car:",
    layout="wide",
)

CURRENT_YEAR = 2024
DEFAULT_BENEFITS = 0
DEFAULT_DISCOUNT_PRICE = 0

BRAND_LABELS = {
    "Bmw": "BMW",
    "Mg": "MG",
    "Maruti": "Maruti Suzuki",
}

LUXURY_BRANDS = ["Audi", "Bmw", "Mercedes Benz", "Jaguar", "Porsche", "Volvo", "Land Rover"]
PREMIUM_BRANDS = ["Mg", "Jeep", "Kia", "Skoda", "Volkswagen", "Toyota"]


st.markdown(
    """
<style>
    :root {
        --c24-bg: #0b0f17;
        --c24-panel: #111827;
        --c24-panel-soft: #172033;
        --c24-line: rgba(148, 163, 184, 0.20);
        --c24-text: #eef2f7;
        --c24-muted: #94a3b8;
        --c24-accent: #ff4d2e;
        --c24-accent-2: #f9b233;
        --c24-green: #21c77a;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 8%, rgba(255, 77, 46, 0.18), transparent 32%),
            linear-gradient(135deg, #0b0f17 0%, #111827 54%, #141414 100%);
        color: var(--c24-text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .premium-hero {
        border: 1px solid var(--c24-line);
        background:
            linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(23, 32, 51, 0.92)),
            linear-gradient(90deg, rgba(255, 77, 46, 0.16), rgba(249, 178, 51, 0.10));
        border-radius: 8px;
        padding: 30px 34px;
        margin-bottom: 22px;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.30);
    }

    .eyebrow {
        color: var(--c24-accent-2);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .premium-hero h1 {
        color: var(--c24-text);
        font-size: clamp(2rem, 4vw, 3.8rem);
        line-height: 1.04;
        letter-spacing: 0;
        margin: 0;
        max-width: 760px;
    }

    .premium-hero p {
        color: var(--c24-muted);
        margin: 14px 0 0 0;
        max-width: 720px;
        font-size: 1rem;
        line-height: 1.6;
    }

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 22px;
    }

    .metric-tile {
        border: 1px solid var(--c24-line);
        background: rgba(255, 255, 255, 0.035);
        border-radius: 8px;
        padding: 14px 16px;
    }

    .metric-tile strong {
        display: block;
        color: var(--c24-text);
        font-size: 1.15rem;
        margin-bottom: 2px;
    }

    .metric-tile span {
        color: var(--c24-muted);
        font-size: 0.82rem;
    }

    .section-title {
        color: var(--c24-text);
        font-size: 1.05rem;
        font-weight: 800;
        margin: 0 0 12px 0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--c24-line);
        background: rgba(17, 24, 39, 0.82);
        border-radius: 8px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="slider"] {
        border-radius: 8px;
    }

    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stTextInput label {
        color: #dbe4ef !important;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        height: 3.2rem;
        border: 0;
        border-radius: 8px;
        color: white;
        font-weight: 800;
        background: linear-gradient(135deg, var(--c24-accent), #d9361e);
        box-shadow: 0 18px 34px rgba(255, 77, 46, 0.22);
    }

    .stButton > button:hover {
        border: 0;
        color: white;
        background: linear-gradient(135deg, #ff6247, #ec3f22);
    }

    .prediction-box {
        border: 1px solid rgba(33, 199, 122, 0.34);
        background: linear-gradient(135deg, rgba(33, 199, 122, 0.15), rgba(17, 24, 39, 0.92));
        border-radius: 8px;
        padding: 22px 24px;
        margin-top: 16px;
    }

    .prediction-box span {
        display: block;
        color: #a7f3d0;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        font-size: 0.76rem;
        margin-bottom: 4px;
    }

    .prediction-box h2 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: 0;
    }

    .prediction-box p {
        color: #cbd5e1;
        margin: 8px 0 0 0;
    }

    @media (max-width: 760px) {
        .premium-hero {
            padding: 24px 20px;
        }

        .metric-strip {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if os.path.exists("xgb_model.pkl"):
        return joblib.load("xgb_model.pkl")
    return None


@st.cache_resource
def load_model_columns():
    if os.path.exists("model_columns.pkl"):
        return joblib.load("model_columns.pkl")
    return []


@st.cache_data
def load_reference_data():
    df = pd.read_csv("cars24_data.csv")
    df["bodytype"] = df["bodytype"].astype(str).str.strip().str.title()
    df["bodytype"] = df["bodytype"].replace({"Suv": "Suv"})
    df["model"] = df["model"].astype(str).str.strip()
    df["make"] = df["make"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()
    df["fueltype"] = df["fueltype"].astype(str).str.strip()
    df["transmission"] = df["transmission"].astype(str).str.strip()
    return df


def display_brand(make):
    return BRAND_LABELS.get(make, make)


def display_body(bodytype):
    return "SUV" if bodytype == "Suv" else bodytype


def get_brand_tier(make):
    if make in LUXURY_BRANDS:
        return "Luxury"
    if make in PREMIUM_BRANDS:
        return "Premium"
    return "Economy"


def top_models_by_make(df):
    top_models = df["model"].value_counts().head(20).index.tolist()
    top_df = df[df["model"].isin(top_models)]
    mapping = {
        make: sorted(group["model"].dropna().unique().tolist())
        for make, group in top_df.groupby("make")
    }
    return mapping, top_models


def model_body_mapping(df, top_models):
    model_body = {}
    for model_name in top_models:
        body_counts = df.loc[
            (df["model"] == model_name) & (df["bodytype"].notna()) & (df["bodytype"] != "Nan"),
            "bodytype",
        ].value_counts()
        if not body_counts.empty:
            model_body[model_name] = body_counts.index[0]
    return model_body


def build_feature_row(model_columns, values):
    car_age = CURRENT_YEAR - values["year"]
    high_mileage = int(values["kilometerdriven"] > values["high_mileage_threshold"])
    km_per_year = round(values["kilometerdriven"] / max(car_age, 1), 2)
    age_km_ratio = round(car_age * values["kilometerdriven"] / 1000, 2)

    row = pd.DataFrame(0, index=[0], columns=model_columns)
    numeric_values = {
        "kilometerdriven": values["kilometerdriven"],
        "ownernumber": values["ownernumber"],
        "transmission": 0 if values["transmission"] == "Automatic" else 1,
        "isc24assured": 0,
        "benefits": DEFAULT_BENEFITS,
        "discountprice": DEFAULT_DISCOUNT_PRICE,
        "car_age": car_age,
        "high_mileage": high_mileage,
        "km_per_year": km_per_year,
        "is_first_owner": int(values["ownernumber"] == 1),
        "age_km_ratio": age_km_ratio,
        "log_km": np.log1p(values["kilometerdriven"]),
    }

    for column, value in numeric_values.items():
        if column in row.columns:
            row.at[0, column] = value

    encoded_values = {
        f"make_{values['make']}": 1,
        f"city_{values['city']}": 1,
        f"fueltype_{values['fueltype']}": 1,
        f"bodytype_{values['bodytype']}": 1,
        f"brand_tier_{get_brand_tier(values['make'])}": 1,
        f"model_clean_{values['model_clean']}": 1,
    }

    for column, value in encoded_values.items():
        if column in row.columns:
            row.at[0, column] = value

    return row


model = load_model()
model_columns = load_model_columns()
data = load_reference_data()
brand_models, top_models = top_models_by_make(data)
model_to_body = model_body_mapping(data, top_models)
high_mileage_threshold = data["kilometerdriven"].quantile(0.75)

makes = sorted(data["make"].dropna().unique().tolist())
cities = sorted(data["city"].dropna().unique().tolist())
fuels = sorted(data["fueltype"].dropna().unique().tolist())
bodytypes = ["Hatchback", "Sedan", "Suv"]
transmissions = sorted(data["transmission"].dropna().unique().tolist())
years = list(range(int(data["year"].max()), int(data["year"].min()) - 1, -1))


st.markdown(
    """
<section class="premium-hero">
    <div class="eyebrow">Cars24 valuation desk</div>
    <h1>Car price prediction</h1>
    <p>
        Choose the brand, model and ownership details. The form keeps models tied
        to the selected company and fills the body type automatically for known cars.
    </p>
    <div class="metric-strip">
        <div class="metric-tile"><strong>20 brands</strong><span>Filtered by company</span></div>
        <div class="metric-tile"><strong>Auto body type</strong><span>Model-aware selection</span></div>
        <div class="metric-tile"><strong>XGBoost</strong><span>Saved model prediction</span></div>
    </div>
</section>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.4, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<p class="section-title">Vehicle details</p>', unsafe_allow_html=True)
        top_col1, top_col2 = st.columns(2)

        with top_col1:
            make = st.selectbox("Company", makes, format_func=display_brand)

        available_models = brand_models.get(make, [])
        model_options = available_models + ["Other"]

        with top_col2:
            model_name = st.selectbox("Car model", model_options)

        resolved_body = model_to_body.get(model_name)
        body_col, year_col = st.columns(2)

        with body_col:
            if resolved_body:
                bodytype = resolved_body
                st.text_input("Body type", value=display_body(bodytype), disabled=True)
            else:
                bodytype = st.selectbox("Body type", bodytypes, format_func=display_body)

        with year_col:
            year = st.selectbox("Manufacturing year", years, index=min(3, len(years) - 1))

        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            fuel = st.selectbox("Fuel type", fuels)
        with spec_col2:
            transmission = st.selectbox("Transmission", transmissions)

        use_col1, use_col2 = st.columns(2)
        with use_col1:
            owner = st.selectbox("Owner number", [1, 2, 3], format_func=lambda value: f"{value} owner")
        with use_col2:
            city = st.selectbox("City", cities, index=cities.index("Mumbai") if "Mumbai" in cities else 0)

        km = st.slider("Kilometers driven", 500, 200000, 30000, step=500)

with right:
    with st.container(border=True):
        st.markdown('<p class="section-title">Selection summary</p>', unsafe_allow_html=True)
        st.metric("Company", display_brand(make))
        st.metric("Model", model_name)
        st.metric("Body", display_body(bodytype))
        st.metric("Kilometers", f"{km:,} km")

        predict = st.button("Predict Price", type="primary")

if predict:
    if model is None or not model_columns:
        st.error("Model files are missing. Please make sure xgb_model.pkl and model_columns.pkl are available.")
    else:
        model_clean = model_name if model_name in top_models else "Other"
        input_row = build_feature_row(
            model_columns,
            {
                "make": make,
                "model_clean": model_clean,
                "city": city,
                "year": year,
                "fueltype": fuel,
                "kilometerdriven": km,
                "ownernumber": owner,
                "transmission": transmission,
                "bodytype": bodytype,
                "high_mileage_threshold": high_mileage_threshold,
            },
        )
        predicted_log_price = model.predict(input_row)[0]
        predicted_price = int(np.expm1(predicted_log_price))

        st.markdown(
            f"""
<div class="prediction-box">
    <span>Estimated selling price</span>
    <h2>Rs. {predicted_price:,.0f}</h2>
    <p>{display_brand(make)} {model_name} | {year} | {display_body(bodytype)} | {km:,} km</p>
</div>
""",
            unsafe_allow_html=True,
        )
