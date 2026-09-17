"""
A simple Streamlit app with input fields for the 12 key house
features, a button to request a prediction, and an area showing the
result. Talks to the FastAPI model API over HTTP.

The API URL is configurable via the API_URL environment variable so
the same code works both in Docker (where the API is reached by its
docker-compose service name, e.g. http://api:8000) and locally
(http://localhost:8000).
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"

NEIGHBORHOOD_OPTIONS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "Names", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]
KITCHEN_QUAL_OPTIONS = ["Ex", "Gd", "TA", "Fa", "Po"]
HOUSE_STYLE_OPTIONS = [
    "1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl",
]
CENTRAL_AIR_OPTIONS = ["Y", "N"]

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write(
    "Enter the key characteristics of a house to get a predicted sale price. "
    "Every other feature is filled in automatically with typical values "
    "from the training data."
)

st.subheader("House details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall quality (1=Poor, 10=Excellent)", 1, 10, 5)
    gr_liv_area = st.number_input(
        "Above-grade living area (sq ft)", min_value=1, value=1500, step=10
    )
    garage_cars = st.number_input("Garage size (car capacity)", 0, 5, 1)
    total_bsmt_sf = st.number_input(
        "Total basement area (sq ft)", min_value=0, value=800, step=10
    )
    full_bath = st.number_input("Full bathrooms above grade", 0, 5, 2)
    bedroom_abv_gr = st.number_input("Bedrooms above grade", 0, 15, 3)

with col2:
    year_built = st.number_input("Year built", min_value=1800, max_value=2100, value=2000)
    lot_area = st.number_input("Lot area (sq ft)", min_value=1, value=8000, step=100)
    neighborhood = st.selectbox("Neighborhood", NEIGHBORHOOD_OPTIONS, index=NEIGHBORHOOD_OPTIONS.index("CollgCr"))
    kitchen_qual = st.selectbox("Kitchen quality", KITCHEN_QUAL_OPTIONS, index=1)
    house_style = st.selectbox("House style", HOUSE_STYLE_OPTIONS, index=3)
    central_air = st.selectbox("Central air conditioning", CENTRAL_AIR_OPTIONS, index=0)

st.divider()

if st.button("Predict price", type="primary"):
    payload = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "Neighborhood": neighborhood,
        "BedroomAbvGr": bedroom_abv_gr,
        "LotArea": lot_area,
        "KitchenQual": kitchen_qual,
        "HouseStyle": house_style,
        "CentralAir": central_air,
    }

    try:
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        predicted_price = response.json()["predicted_price"]
        st.success(f"### Predicted sale price: ${predicted_price:,.0f}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the model API at '{API_URL}'. "
            "Make sure the API container is running."
        )
    except requests.exceptions.HTTPError as exc:
        st.error(f"The API returned an error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
