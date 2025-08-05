import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os

# Get current directory
base_path = os.path.dirname(__file__)

# Load model
model_path = os.path.join(base_path, 'pipe_model.joblib')
ml_model = joblib.load(model_path)

# Load frequency map (corrected filename)
freq_path = os.path.join(base_path, 'car_freq_map.pkl')
freq_map = joblib.load(open(freq_path, 'rb'))

companies = {
    "Maruti Suzuki": [
        "ritz", "sx4", "ciaz", "wagon r", "swift", "vitara brezza", "s cross", "alto 800",
        "ertiga", "dzire", "alto k10", "ignis", "800", "baleno", "omni"
    ],
    "Toyota": [
        "fortuner", "innova", "corolla altis", "etios cross", "etios g", "etios liva",
        "corolla", "etios gd", "camry", "land cruiser"
    ],
    "Royal Enfield": [
        "Royal Enfield Thunder 500", "Royal Enfield Classic 350", "Royal Enfield Thunder 350",
        "Royal Enfield Bullet 350", "Royal Enfield Classic 500"
    ],
    "UM": ["UM Renegade Mojave"],
    "KTM": ["KTM RC200", "KTM RC390", "KTM 390 Duke "],
    "Bajaj": [
        "Bajaj Dominar 400", "Bajaj Pulsar RS200", "Bajaj Avenger 220", "Bajaj Avenger 150",
        "Bajaj Avenger 220 dtsi", "Bajaj Avenger 150 street", "Bajaj Pulsar  NS 200",
        "Bajaj Pulsar 220 F", "Bajaj Pulsar NS 200", "Bajaj Pulsar 135 LS", "Bajaj Discover 100",
        "Bajaj Discover 125", "Bajaj  ct 100", "Bajaj Avenger Street 220"
    ],
    "Honda": [
        "Honda CB Hornet 160R", "Honda CBR 150", "Honda Activa 4G", "Honda Dream Yuga ",
        "Honda CB Trigger", "Honda CB Unicorn", "Honda Karizma", "Honda Activa 125",
        "Honda CB Shine", "Honda CB twister"
    ],
    "Yamaha": [
        "Yamaha FZ S V 2.0", "Yamaha FZ 16", "Yamaha FZ  v 2.0", "Yamaha Fazer ", "Yamaha FZ S "
    ],
    "TVS": [
        "TVS Apache RTR 160", "TVS Apache RTR 180", "TVS Sport ", "TVS Jupyter", "TVS Wego"
    ],
    "Hero": [
        "Hero Extreme", "Hero Passion X pro", "Hero Splender iSmart", "Hero Passion Pro",
        "Hero Honda CBZ extreme", "Hero Honda Passion Pro", "Hero Splender Plus", "Hero Glamour",
        "Hero Super Splendor", "Hero Hunk", "Hero  Ignitor Disc", "Hero  CBZ Xtreme"
    ],
    "Hyundai": [
        "i20", "grand i10", "i10", "eon", "xcent", "elantra", "creta", "verna",
        "city", "brio", "amaze", "jazz"
    ]
}
model_images = {
    "ritz": "vehicle_image/ritz.jpeg",
    "sx4": "vehicle_image/sx4.jpeg",
    "ciaz": "vehicle_image/ciaz.jpeg",
    "wagon r": "vehicle_image/wagon_r.jpeg",
    "swift": "vehicle_image/swift.jpeg",
    "vitara brezza": "vehicle_image/brezza.jpeg",
    "s cross": "vehicle_image/scross.jpeg",
    "alto 800": "vehicle_image/alto800.jpeg",
    "ertiga": "vehicle_image/ertiga.jpeg",
    "dzire": "vehicle_image/dzier.jpeg",
    "alto k10": "vehicle_image/800.jpeg",
    "ignis": "vehicle_image/ignis.jpeg",
    "800": "vehicle_image/800.jpeg",
    "baleno": "vehicle_image/baleno.jpeg",
    "omni": "vehicle_image/omni.jpeg",

    # Toyota
    "fortuner": "vehicle_image/fortuner.jpeg",
    "innova": "vehicle_image/innova.jpeg",
    "corolla altis": "vehicle_image/corolla.jpeg",
    "etios cross": "vehicle_image/etios_cross.jpeg",
    "etios g": "vehicle_image/etios_g.jpeg",
    "etios liva": "vehicle_image/etios_liva.jpeg",
    "corolla": "vehicle_image/corolla.jpeg",
    "etios gd": "vehicle_image/etios_gd.jpeg",
    "camry": "vehicle_image/camry.jpeg",
    "land cruiser": "vehicle_image/land_cruiser.jpeg",

    # Royal Enfield
    "royal enfield thunder 500": "vehicle_image/Royal Enfield Thunder 500.jpeg",
    "royal enfield classic 350": "vehicle_image/Royal Enfield Classic 350.jpeg",
    "royal enfield thunder 350": "vehicle_image/Royal Enfield Thunder 350.jpeg",
    "royal enfield bullet 350": "vehicle_image/Royal Enfield Bullet 350.jpeg",
    "royal enfield classic 500": "vehicle_image/Royal Enfield Classic 500.jpeg",

    # UM
    "um renegade mojave": "vehicle_image/UM Renegade Mojave.jpeg",

    # KTM
    "ktm rc200": "vehicle_image/KTM RC200.jpeg",
    "ktm rc390": "vehicle_image/KTM RC390.jpeg",
    "ktm 390 duke": "vehicle_image/KTM 390 Duke.jpeg",

    # Bajaj
    "bajaj dominar 400": "vehicle_image/Bajaj Dominar 400.jpeg",
    "bajaj pulsar rs200": "vehicle_image/Bajaj Pulsar RS200.jpeg",
    "bajaj avenger 220": "vehicle_image/Bajaj Avenger 220.jpeg",
    "bajaj avenger 150": "vehicle_image/Bajaj Avenger 150.jpeg",
    "bajaj avenger 220 dtsi": "vehicle_image/Bajaj Avenger 220 dtsi.jpeg",
    "bajaj avenger 150 street": "vehicle_image/Bajaj Avenger 150 street.jpeg",
    "bajaj pulsar ns 200": "vehicle_image/Bajaj Pulsar NS 200.jpeg",
    "bajaj pulsar 220 f": "vehicle_image/Bajaj Pulsar 220 F.jpeg",
    "bajaj pulsar 135 ls": "vehicle_image/Bajaj Pulsar 135 LS.jpeg",
    "bajaj discover 100": "vehicle_image/Bajaj Discover 100.jpeg",
    "bajaj discover 125": "vehicle_image/Bajaj Discover 125.jpeg",
    "bajaj ct 100": "vehicle_image/Bajaj ct 100.jpeg",
    "bajaj avenger street 220": "vehicle_image/Bajaj Avenger Street 220.jpeg",

    # Honda
    "honda cb hornet 160r": "vehicle_image/Honda CB Hornet 160R.jpeg",
    "honda cbr 150": "vehicle_image/Honda CBR 150.jpeg",
    "honda activa 4g": "vehicle_image/Honda Activa 4G.jpeg",
    "honda dream yuga": "vehicle_image/Honda dream yuga.jpeg",
    "honda cb trigger": "vehicle_image/Honda CB Trigger.jpeg",
    "honda cb unicorn": "vehicle_image/Honda CB Unicorn.jpeg",
    "honda karizma": "vehicle_image/Honda Karizma.jpeg",
    "honda activa 125": "vehicle_image/Honda Activa 125.jpeg",
    "honda cb shine": "vehicle_image/Honda CB Shine.jpeg",
    "honda cb twister": "vehicle_image/Honda CB twister.jpeg",

    # Yamaha
    "yamaha fz s v 2.0": "vehicle_image/Yamaha FZ S V 2.0.jpeg",
    "yamaha fz 16": "vehicle_image/Yamaha FZ 16.jpeg",
    "yamaha fz v 2.0": "vehicle_image/Yamaha FZ V 2.0.jpeg",
    "yamaha fazer": "vehicle_image/Yamaha Fazer.jpeg",
    "yamaha fz s": "vehicle_image/Yamaha FZ S.jpeg",

    # TVS
    "tvs apache rtr 160": "vehicle_image/TVS Apache RTR 160.jpeg",
    "tvs apache rtr 180": "vehicle_image/TVS Apache RTR 180.jpeg",
    "tvs sport": "vehicle_image/TVS Sport.jpeg",
    "tvs jupyter": "vehicle_image/TVS Jupyter.jpeg",
    "tvs wego": "vehicle_image/TVS Wego.jpeg",

    # Hero
    "hero extreme": "vehicle_image/Hero Extreme.jpeg",
    "hero passion x pro": "vehicle_image/Hero Passion X pro.jpeg",
    "hero splender ismart": "vehicle_image/Hero Splender iSmart.jpeg",
    "hero passion pro": "vehicle_image/Hero Passion Pro.jpeg",
    "hero honda cbz extreme": "vehicle_image/Hero Honda CBZ extreme.jpeg",
    "hero honda passion pro": "vehicle_image/Hero Honda Passion Pro.jpeg",
    "hero splender plus": "vehicle_image/Hero Splender Plus.jpeg",
    "hero glamour": "vehicle_image/Hero Glamour.jpeg",
    "hero super splendor": "vehicle_image/Hero Super Splendor.jpeg",
    "hero hunk": "vehicle_image/Hero Hunk.jpeg",
    "hero ignitor disc": "vehicle_image/Hero Ignitor Disc.jpeg",
    "hero cbz xtreme": "vehicle_image/Hero CBZ Xtreme.jpeg",

    # Hyundai
    "i20": "vehicle_image/i20.jpeg",
    "grand i10": "vehicle_image/grand i10.jpeg",
    "i10": "vehicle_image/i10.jpeg",
    "eon": "vehicle_image/eon.jpeg",
    "xcent": "vehicle_image/xcent.jpeg",
    "elantra": "vehicle_image/elantra.jpeg",
    "creta": "vehicle_image/creta.jpeg",
    "verna": "vehicle_image/verna.jpeg",
    "city": "vehicle_image/city.jpeg",
    "brio": "vehicle_image/brio.jpeg",
    "amaze": "vehicle_image/amaze.jpeg",
    "jazz": "vehicle_image/jazz.jpeg"
}

st.title("Second Hand Vehicle Price Predictor")

# Layout with two columns
with st.container():
    left_column, right_column = st.columns([1, 2]) 

with right_column:
    company = st.selectbox("Select Company", options=list(companies.keys()))
    models = companies[company]
    model = st.selectbox("Select Model", options=models).strip().lower()

    Driven_km = st.number_input("Driven kilometers", min_value=0, max_value=500000, value=1000, step=100)
    Year = st.number_input("Year", min_value=1990, max_value=2025, value=2018, step=1)
    Present_Price = st.number_input("Present_Price (e.g. 5.35 lakhs)", min_value=0.0, step=0.1)

    Fuel_Type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])
    Transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
    Seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer"])
    Owner = st.selectbox("Owner (0: First, 1: Second, 2: Third)", options=[0, 1, 2])

car_model_freq = freq_map.get(model, 0)

input_df = pd.DataFrame([{
    "Car_Name": car_model_freq,           # frequency-encoded car name
    "Year": Year,
    "Present_Price": Present_Price,
    "Driven_kms": Driven_km,
    "Fuel_Type": Fuel_Type,
    "Selling_type": Seller_type,
    "Transmission": Transmission,
    "Owner": Owner
}])
if st.button("Predict Price"):
    prediction = ml_model.predict(input_df)[0]
    st.success(f"💰Predicted Price: ₹{round(prediction, 2)} lakhs")


# Left side image display
with left_column:
    relative_path = model_images[model]
    image_path = os.path.join(base_path, relative_path)
    
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image , use_container_width=True)
