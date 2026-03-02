import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Japan Real Estate Price Predictor",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Icon helper ---------- */
.ic {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    vertical-align: middle;
    flex-shrink: 0;
}
.ic svg {
    fill: currentColor;
    width: 1em;
    height: 1em;
}

/* ---------- Global ---------- */
html, body, .stApp {
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}

/* ---------- Hero Banner ---------- */
.hero {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    padding: 2.4rem 2.8rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
}
.hero-row {
    display: flex;
    align-items: center;
    gap: 16px;
}
.hero-icon {
    color: #5b8def;
    font-size: 40px;
}
.hero-icon svg {
    width: 40px;
    height: 40px;
    fill: #5b8def;
}
.hero h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e8e8f0;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 0.95rem;
    color: #8888a8;
    margin: 0.5rem 0 0;
    max-width: 560px;
    line-height: 1.55;
}

/* ---------- Prediction Card ---------- */
.pred-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 2.4rem 2rem;
    text-align: center;
}
.pred-card-top {
    width: 100%;
    height: 3px;
    background: #5b8def;
    border-radius: 3px;
    margin-bottom: 1.6rem;
}
.pred-label {
    font-size: 0.82rem;
    color: #8888a8;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 600;
}
.pred-price {
    font-size: 2.8rem;
    font-weight: 700;
    color: #5b8def;
    margin: 0.6rem 0 0.2rem;
}
.pred-usd {
    font-size: 1rem;
    color: #5a5a78;
}

/* ---------- Metric Mini-Card ---------- */
.mcard {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 0.8rem;
    text-align: center;
}
.mcard .m-icon {
    font-size: 24px;
    color: #5b8def;
    margin-bottom: 0.3rem;
}
.mcard .m-icon svg {
    width: 24px;
    height: 24px;
    fill: #5b8def;
}
.mcard .value {font-size: 1.3rem; font-weight: 600; color: #d0d0e0;}
.mcard .label {font-size: 0.75rem; color: #6a6a88; margin-top: 0.15rem;}

/* ---------- Sidebar ---------- */
.sidebar-title {
    text-align: center;
    padding: 0.8rem 0 0.4rem;
}
.sidebar-title .s-icon {
    color: #5b8def;
    font-size: 36px;
}
.sidebar-title .s-icon svg {
    width: 36px;
    height: 36px;
    fill: #5b8def;
}
.sidebar-title h2 {margin: 0.3rem 0 0; font-weight: 600; font-size: 1.15rem; color: #d0d0e0;}
.sidebar-title p {color: #6a6a88; font-size: 0.8rem; margin: 0;}

.sb-header {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #5b8def;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2a2a4a;
}
.sb-header .ic {font-size: 18px; color: #5b8def;}
.sb-header .ic svg {width: 18px; height: 18px; fill: #5b8def;}

/* ---------- Info box ---------- */
.info-box {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin: 1rem 0;
    line-height: 1.65;
    color: #c0c0d8;
}
.info-box h3 {color: #d0d0e0;}

/* ---------- Section Header ---------- */
.sec-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #d0d0e0;
    margin: 1.4rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2a2a4a;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-header .ic {font-size: 22px; color: #5b8def;}
.sec-header .ic svg {width: 22px; height: 22px; fill: #5b8def;}

/* ---------- Buttons ---------- */
div.stButton > button {
    background: #5b8def !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: background 0.2s ease !important;
}
div.stButton > button:hover {
    background: #4a7bde !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px; font-weight: 500;}
</style>
""",
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA LOADING & PREPROCESSING  (cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner="Loading model & data …")
def load_pipeline():
    """Rebuild the exact same preprocessing pipeline from the notebook."""

    data = pd.read_csv("02.csv", low_memory=False)

    drop_cols = [
        "No", "UnitPrice", "PricePerTsubo", "Period", "Remarks",
        "Renovation", "FloorPlan", "Purpose", "Use", "MunicipalityCode",
        "TimeToNearestStation", "MaxTimeToNearestStation",
        "FrontageIsGreaterFlag", "AreaIsGreaterFlag",
        "TotalFloorAreaIsGreaterFlag", "Prefecture",
    ]
    data.drop(columns=drop_cols, inplace=True)

    data = data[data["Type"] != "Agricultural Land"].copy()

    na_cols = [
        "Region", "DistrictName", "NearestStation", "MinTimeToNearestStation",
        "LandShape", "Frontage", "TotalFloorArea", "BuildingYear", "Structure",
        "Classification", "Breadth", "CityPlanning", "CoverageRatio", "FloorAreaRatio",
    ]
    data.dropna(subset=na_cols, inplace=True)
    data.dropna(subset=["Direction"], inplace=True)

    cat_cols = [
        "Type", "Region", "Municipality", "DistrictName", "NearestStation",
        "LandShape", "Structure", "Classification", "CityPlanning", "Direction",
    ]
    num_cols = [
        "Frontage", "TotalFloorArea", "BuildingYear", "Breadth",
        "CoverageRatio", "FloorAreaRatio", "MinTimeToNearestStation", "Area",
    ]

    unique_values = {}
    for col in cat_cols:
        unique_values[col] = sorted(data[col].dropna().unique().tolist())

    analysis_data = data.copy()

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    scaler1 = MinMaxScaler()
    data[num_cols] = scaler1.fit_transform(data[num_cols])

    data["AgeOfBuilding"] = data["Year"] - data["BuildingYear"]

    num_cols_ext = num_cols + ["AgeOfBuilding"]
    scaler2 = MinMaxScaler()
    data[num_cols_ext] = scaler2.fit_transform(data[num_cols_ext])

    feature_cols = [c for c in data.columns if c != "TradePrice"]

    model = joblib.load("rf_model_new.joblib")

    return {
        "model": model,
        "label_encoders": label_encoders,
        "scaler1": scaler1,
        "scaler2": scaler2,
        "num_cols": num_cols,
        "num_cols_ext": num_cols_ext,
        "cat_cols": cat_cols,
        "unique_values": unique_values,
        "feature_cols": feature_cols,
        "analysis_data": analysis_data,
    }


def predict_price(pipe, raw_input: dict) -> float:
    """Transform a single raw-input dict and return predicted price."""

    df = pd.DataFrame([raw_input])

    for col in pipe["cat_cols"]:
        df[col] = pipe["label_encoders"][col].transform(df[col].astype(str))

    df[pipe["num_cols"]] = pipe["scaler1"].transform(df[pipe["num_cols"]])
    df["AgeOfBuilding"] = df["Year"] - df["BuildingYear"]
    df[pipe["num_cols_ext"]] = pipe["scaler2"].transform(df[pipe["num_cols_ext"]])
    df = df[pipe["feature_cols"]]

    log_price = pipe["model"].predict(df)
    return float(np.expm1(log_price)[0])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INITIALISE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pipe = load_pipeline()
uv = pipe["unique_values"]
analysis = pipe["analysis_data"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HERO BANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SVG icons (from Material Design)
ICON_HOUSE = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M220-180h150v-250h220v250h150v-390L480-765 220-570v390Zm-60 60v-480l320-240 320 240v480H530v-250H430v250H160Zm320-353Z"/></svg>'
ICON_COTTAGE = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M160-200v-319L80-452l-42-58 442-332 200 150v-108h120v228l162 122-42 58-80-60v241H520v-240H440v240H160Zm60-60h160v-240h200v240h160v-284L480-723 220-544v284Zm200-314q25 0 42.5-17.5T480-634q0-25-17.5-42.5T420-694q-25 0-42.5 17.5T360-634q0 25 17.5 42.5T420-574Zm60 114Z"/></svg>'
ICON_SELL = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="m570-104-56-56 124-124H200v-480h80v400h358L514-488l56-56 220 220-220 220Z"/></svg>'
ICON_LOCATION = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M480-480q33 0 56.5-23.5T560-560q0-33-23.5-56.5T480-640q-33 0-56.5 23.5T400-560q0 33 23.5 56.5T480-480Zm0 294q122-112 181-203.5T720-552q0-109-69.5-178.5T480-800q-101 0-170.5 69.5T240-552q0 71 59 162.5T480-186Zm0 106Q319-217 239.5-334.5T160-552q0-150 96.5-239T480-880q127 0 223.5 89T800-552q0 100-79.5 217.5T480-80Zm0-480Z"/></svg>'
ICON_RULER = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm40-80h200v-60H240v60Zm287-160 113-113-28-29-85 85-43-43 85-85-28-29-85 85-43-43 85-85-28-28-113 113 170 172ZM240-560h80v-160h-80v160Zm-40 360v-560 560Z"/></svg>'
ICON_BUILDING = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M120-120v-555l240-165 240 165v75h240v480H120Zm80-80h80v-80h-80v80Zm0-160h80v-80h-80v80Zm0-160h80v-80h-80v80Zm160 320h80v-80h-80v80Zm0-160h80v-80h-80v80Zm0-160h80v-80h-80v80Zm160 320h80v-80h-80v80Zm0-160h80v-80h-80v80Zm160 160h80v-80h-80v80Zm0-160h80v-80h-80v80Z"/></svg>'
ICON_MAP = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="m600-120-240-84-186 72q-20 8-37-4.5T120-170v-560q0-13 7.5-23t20.5-15l212-72 240 84 186-72q20-8 37 4.5t17 33.5v560q0 13-7.5 23T812-192l-212 72Zm-40-98v-468l-160-56v468l160 56Zm80 0 120-40v-474l-120 46v468Zm-400-10 120-46v-468l-120 40v474Zm400-458v468-468Zm-280-56v468-468Z"/></svg>'
ICON_CALENDAR = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M200-80q-33 0-56.5-23.5T120-160v-560q0-33 23.5-56.5T200-800h40v-80h80v80h320v-80h80v80h40q33 0 56.5 23.5T840-720v560q0 33-23.5 56.5T760-80H200Zm0-80h560v-400H200v400Zm0-480h560v-80H200v80Zm0 0v-80 80Z"/></svg>'
ICON_CROP = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M200-680v400h400v80H120v-480h80Zm160-160h480v480h-80v-400H360v-80Z"/></svg>'
ICON_HOME = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M220-180h150v-250h220v250h150v-390L480-765 220-570v390Zm-60 60v-480l320-240 320 240v480H530v-250H430v250H160Zm320-353Z"/></svg>'
ICON_EVENT = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M580-240q-42 0-71-29t-29-71q0-42 29-71t71-29q42 0 71 29t29 71q0 42-29 71t-71 29ZM200-80q-33 0-56.5-23.5T120-160v-560q0-33 23.5-56.5T200-800h40v-80h80v80h320v-80h80v80h40q33 0 56.5 23.5T840-720v560q0 33-23.5 56.5T760-80H200Zm0-80h560v-400H200v400Z"/></svg>'
ICON_WALK = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="m280-40 112-564-72 28v136h-80v-188l202-86q14-6 29.5-7t29.5 4q14 5 26.5 14t20.5 23l40 64q26 42 70.5 69T760-520v80q-66 0-123.5-27.5T540-540l-24 120 84 80v300h-80v-240l-84-80-72 320h-84Zm260-700q-33 0-56.5-23.5T460-820q0-33 23.5-56.5T540-900q33 0 56.5 23.5T620-820q0 33-23.5 56.5T540-740Z"/></svg>'
ICON_CHART = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M640-160v-280h160v280H640Zm-240 0v-640h160v640H400Zm-240 0v-440h160v440H160Z"/></svg>'
ICON_INSIGHTS = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M120-120v-80l80-80v160h-80Zm160 0v-240l80-80v320h-80Zm160 0v-320l80 81v239h-80Zm160 0v-239l80-80v319h-80Zm160 0v-400l80-80v480h-80ZM120-327v-113l280-280 160 160 280-280v113L560-447 400-607 120-327Z"/></svg>'
ICON_INFO = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M440-280h80v-240h-80v240Zm40-320q17 0 28.5-11.5T520-640q0-17-11.5-28.5T480-680q-17 0-28.5 11.5T440-640q0 17 11.5 28.5T480-600ZM480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z"/></svg>'

st.markdown(
    f"""
<div class="hero">
    <div class="hero-row">
        <span class="hero-icon">{ICON_HOUSE}</span>
        <h1>Japan Real Estate Price Predictor</h1>
    </div>
    <p>Property valuation powered by a Random Forest model trained on
    50,000+ Japanese real-estate transactions.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR – INPUT FORM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-title">'
        f'<span class="s-icon">{ICON_COTTAGE}</span>'
        "<h2>Property Details</h2>"
        "<p>Configure the property to estimate</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(f'<div class="sb-header"><span class="ic">{ICON_SELL}</span> Property Type</div>', unsafe_allow_html=True)
    in_type = st.selectbox("Type", uv["Type"])
    in_region = st.selectbox("Region", uv["Region"])

    st.markdown(f'<div class="sb-header"><span class="ic">{ICON_LOCATION}</span> Location</div>', unsafe_allow_html=True)
    in_municipality = st.selectbox("Municipality", uv["Municipality"])
    in_district = st.selectbox("District Name", uv["DistrictName"])
    in_station = st.selectbox("Nearest Station", uv["NearestStation"])
    in_time = st.slider("Walk to Station (min)", 0, 120, 10)

    st.markdown(f'<div class="sb-header"><span class="ic">{ICON_RULER}</span> Dimensions</div>', unsafe_allow_html=True)
    in_area = st.number_input("Land Area (m\u00b2)", 10.0, 5000.0, 200.0, step=10.0)
    in_floor = st.number_input("Total Floor Area (m\u00b2)", 10.0, 2000.0, 120.0, step=10.0)
    in_front = st.number_input("Frontage (m)", 0.5, 50.0, 10.0, step=0.5)
    in_shape = st.selectbox("Land Shape", uv["LandShape"])

    st.markdown(f'<div class="sb-header"><span class="ic">{ICON_BUILDING}</span> Building</div>', unsafe_allow_html=True)
    in_byear = st.number_input("Building Year", 1945, 2020, 2010, step=1)
    in_prewar = st.selectbox("Pre-war Building", [0, 1], format_func=lambda x: "Yes" if x else "No")
    in_struct = st.selectbox("Structure", uv["Structure"],
                             help="W = Wood, S = Steel, RC = Reinforced Concrete, SRC = Steel RC")
    in_dir = st.selectbox("Direction", uv["Direction"])

    st.markdown(f'<div class="sb-header"><span class="ic">{ICON_MAP}</span> Zoning & Roads</div>', unsafe_allow_html=True)
    in_class = st.selectbox("Road Classification", uv["Classification"])
    in_breadth = st.number_input("Road Width (m)", 1.0, 60.0, 6.0, step=0.5)
    in_city = st.selectbox("City Planning Zone", uv["CityPlanning"])
    in_cover = st.number_input("Coverage Ratio (%)", 30.0, 80.0, 60.0, step=5.0)
    in_far = st.number_input("Floor-Area Ratio (%)", 50.0, 800.0, 200.0, step=10.0)

    st.markdown(f'<div class="sb-header"><span class="ic">{ICON_CALENDAR}</span> Transaction Period</div>', unsafe_allow_html=True)
    in_year = st.number_input("Transaction Year", 2006, 2019, 2018, step=1)
    in_quarter = st.selectbox("Quarter", [1, 2, 3, 4])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREDICTION BUTTON  &  RESULT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gap1, btn_col, gap2 = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("Predict Property Price", use_container_width=True)

if predict_btn:
    raw = {
        "Type": in_type,
        "Region": in_region,
        "Municipality": in_municipality,
        "DistrictName": in_district,
        "NearestStation": in_station,
        "MinTimeToNearestStation": in_time,
        "Area": in_area,
        "LandShape": in_shape,
        "Frontage": in_front,
        "TotalFloorArea": in_floor,
        "BuildingYear": in_byear,
        "PrewarBuilding": in_prewar,
        "Structure": in_struct,
        "Direction": in_dir,
        "Classification": in_class,
        "Breadth": in_breadth,
        "CityPlanning": in_city,
        "CoverageRatio": in_cover,
        "FloorAreaRatio": in_far,
        "Year": in_year,
        "Quarter": in_quarter,
    }

    try:
        with st.spinner("Analyzing property..."):
            price = predict_price(pipe, raw)

        st.markdown("---")

        _, card_col, _ = st.columns([1, 2, 1])
        with card_col:
            st.markdown(
                f"""
            <div class="pred-card">
                <div class="pred-card-top"></div>
                <div class="pred-label">Estimated Property Price</div>
                <div class="pred-price">&yen; {price:,.0f}</div>
                <div class="pred-usd">Approx. $ {price / 150:,.0f} USD</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, ICON_CROP, f"{in_area:.0f} m\u00b2", "Land Area"),
            (c2, ICON_HOME, f"{in_floor:.0f} m\u00b2", "Floor Area"),
            (c3, ICON_EVENT, str(in_byear), "Built Year"),
            (c4, ICON_WALK, f"{in_time} min", "To Station"),
        ]
        for col, icon, val, lbl in cards:
            with col:
                st.markdown(
                    f'<div class="mcard">'
                    f'<div class="m-icon">{icon}</div>'
                    f'<div class="value">{val}</div>'
                    f'<div class="label">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        st.info("Make sure the selected categorical values exist in the training data.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("<br>", unsafe_allow_html=True)
tab_market, tab_features, tab_about = st.tabs(
    ["Market Insights", "Feature Analysis", "About"]
)

BLUE = "#5b8def"
SLATE = "#8888a8"
TEAL = "#4db6ac"
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#c0c0d8",
    title_font_size=16,
    margin=dict(t=50, b=30, l=30, r=20),
)

with tab_market:
    st.markdown(
        f'<div class="sec-header"><span class="ic">{ICON_CHART}</span> Market Overview</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average Price", f"\u00a5{analysis['TradePrice'].mean():,.0f}")
    k2.metric("Median Price", f"\u00a5{analysis['TradePrice'].median():,.0f}")
    k3.metric("Total Records", f"{len(analysis):,}")
    k4.metric("Avg Land Area", f"{analysis['Area'].mean():,.0f} m\u00b2")

    left, right = st.columns(2)

    with left:
        capped = analysis[analysis["TradePrice"] < analysis["TradePrice"].quantile(0.95)]
        fig1 = px.histogram(
            capped,
            x="TradePrice",
            nbins=60,
            title="Price Distribution (< 95th percentile)",
            labels={"TradePrice": "Trade Price (\u00a5)"},
            color_discrete_sequence=[BLUE],
        )
        fig1.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        reg = (
            analysis.groupby("Region")["TradePrice"]
            .mean()
            .sort_values(ascending=True)
            .reset_index()
        )
        fig2 = px.bar(
            reg,
            x="TradePrice",
            y="Region",
            orientation="h",
            title="Average Price by Region",
            labels={"TradePrice": "Avg Price (\u00a5)", "Region": ""},
            color_discrete_sequence=[TEAL],
        )
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:
        sample = capped.sample(min(3000, len(capped)), random_state=42)
        fig3 = px.scatter(
            sample,
            x="Area",
            y="TradePrice",
            color="Region",
            opacity=0.5,
            title="Price vs Land Area",
            labels={"Area": "Land Area (m\u00b2)", "TradePrice": "Price (\u00a5)"},
            color_discrete_sequence=["#5b8def", "#4db6ac", "#ef8d5b", "#b05bef"],
        )
        fig3.update_layout(**PLOTLY_LAYOUT, height=430)
        st.plotly_chart(fig3, use_container_width=True)

    with right2:
        struct = (
            analysis.groupby("Structure")["TradePrice"]
            .mean()
            .sort_values(ascending=True)
            .tail(10)
            .reset_index()
        )
        fig4 = px.bar(
            struct,
            x="TradePrice",
            y="Structure",
            orientation="h",
            title="Avg Price by Structure (Top 10)",
            labels={"TradePrice": "Avg Price (\u00a5)", "Structure": ""},
            color_discrete_sequence=[SLATE],
        )
        fig4.update_layout(**PLOTLY_LAYOUT, height=430)
        st.plotly_chart(fig4, use_container_width=True)

with tab_features:
    st.markdown(
        f'<div class="sec-header"><span class="ic">{ICON_INSIGHTS}</span> Feature Analysis</div>',
        unsafe_allow_html=True,
    )

    fi = (
        pd.DataFrame(
            {
                "Feature": pipe["feature_cols"],
                "Importance": pipe["model"].feature_importances_,
            }
        )
        .sort_values("Importance", ascending=True)
    )
    fig5 = px.bar(
        fi.tail(15),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Feature Importances (Random Forest)",
        color_discrete_sequence=[BLUE],
    )
    fig5.update_layout(**PLOTLY_LAYOUT, height=500)
    st.plotly_chart(fig5, use_container_width=True)

    fl, fr = st.columns(2)

    with fl:
        corr_cols = [
            "Area", "TotalFloorArea", "Frontage", "BuildingYear",
            "Breadth", "MinTimeToNearestStation", "CoverageRatio",
            "FloorAreaRatio", "TradePrice",
        ]
        corr = analysis[corr_cols].corr()["TradePrice"].drop("TradePrice").sort_values()
        fig6 = px.bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            title="Correlation with Trade Price",
            labels={"x": "Correlation", "y": ""},
            color_discrete_sequence=[TEAL],
        )
        fig6.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig6, use_container_width=True)

    with fr:
        yr = analysis.groupby("Year")["TradePrice"].agg(["mean", "median"]).reset_index()
        fig7 = go.Figure()
        fig7.add_trace(
            go.Scatter(
                x=yr["Year"], y=yr["mean"],
                name="Mean Price",
                line=dict(color=BLUE, width=2),
            )
        )
        fig7.add_trace(
            go.Scatter(
                x=yr["Year"], y=yr["median"],
                name="Median Price",
                line=dict(color=TEAL, width=2),
            )
        )
        fig7.update_layout(
            title="Price Trend Over Years",
            xaxis_title="Year",
            yaxis_title="Price (\u00a5)",
            height=420,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig7, use_container_width=True)

with tab_about:
    st.markdown(
        f'<div class="sec-header"><span class="ic">{ICON_INFO}</span> About This App</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
        <h3 style="margin-top:0;">Japan Real Estate Price Predictor</h3>
        <p>This application uses a <strong>Random Forest Regressor</strong> trained on
        50,000+ Japanese real-estate transaction records to predict property prices.
        The model considers location, property dimensions, building characteristics,
        and zoning information.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    ab1, ab2 = st.columns(2)

    with ab1:
        st.markdown(
            """
#### Model Details
| | |
|---|---|
| **Algorithm** | Random Forest Regressor |
| **Target** | Trade Price (log-transformed) |
| **Features** | 22 input features |
| **Preprocessing** | Label Encoding + MinMax Scaling |

#### Feature Categories
- **Location** -- Municipality, District, Station, Walking time
- **Property** -- Area, Floor Area, Frontage, Shape
- **Building** -- Year, Structure, Direction
- **Zoning** -- City Planning Zone, Coverage & Floor-Area ratios
"""
        )

    with ab2:
        st.markdown(
            """
#### Data Source
- Japanese real-estate transaction records
- Multiple property types & regions
- Comprehensive feature set covering physical, locational, and regulatory attributes

#### Tech Stack
- **Machine Learning** -- scikit-learn (Random Forest)
- **Frontend** -- Streamlit
- **Visualization** -- Plotly
- **Data** -- Pandas, NumPy
"""
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#5a5a78;padding:0.8rem 0;font-size:0.85rem;">'
    "Japan Real Estate Price Predictor &mdash; Built with Streamlit & scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
