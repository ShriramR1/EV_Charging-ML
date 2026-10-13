import os
import pickle
import pandas as pd
import streamlit as st

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="EV Charging Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== DESIGN =====================
st.markdown("""
<style>
.stApp {
    background: #F3FAF4;
    color: #203629;
}

h1, h2, h3, p, label, .stMarkdown {
    color: #203629;
}

h1 {
    color: #176534;
    font-weight: 800;
}

h2, h3 {
    color: #237442;
}

/* BLACK SIDEBAR */
[data-testid="stSidebar"] {
    background: #000000;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #191919;
    border-radius: 9px;
    padding: 8px;
    margin-bottom: 5px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #287D46;
}

[data-testid="stSidebar"] hr {
    border-color: #444444;
}

/* KPI CARDS */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #C7E6CE;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 3px 12px #174D3010;
    transition: transform .2s ease, box-shadow .2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px #174D3020;
}

[data-testid="stMetricLabel"] {
    color: #42634D !important;
}

[data-testid="stMetricValue"] {
    color: #176534 !important;
}

/* INPUT FIELDS */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #FFFFFF;
    color: #203629;
    border-radius: 9px;
}

[data-testid="stNumberInput"] input {
    border: 1px solid #A8D4B2;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #8ED9A1, #54B875);
    color: #123D22 !important;
    border: none;
    border-radius: 11px;
    padding: 10px 22px;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button:hover {
    background: #258448;
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 14px #25844840;
}

hr {
    border-color: #C7E6CE;
}

/* SUBTLE ANIMATION */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(7px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-testid="stVerticalBlock"] {
    animation: fadeIn .35s ease;
}
</style>
""", unsafe_allow_html=True)


# ===================== LOAD MODEL =====================
@st.cache_resource
def load_pickle():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ev_models.pkl"
    )

    with open(path, "rb") as file:
        return pickle.load(file)


try:
    saved = load_pickle()
except FileNotFoundError:
    st.error("ev_models.pkl not found. Keep it beside app.py.")
    st.stop()

df = saved["eda_data"]

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ===================== FIVE INPUTS =====================
def session_form(key):
    st.subheader("🌿 Charging Session Details")
    st.caption("Enter your five features to predict charging demand.")

    c1, c2 = st.columns(2)

    charger_mean = c1.number_input(
        "1. Previous Average Charger Energy (kWh)",
        min_value=0.0,
        value=15.0,
        step=1.0,
        key=key + "charger_mean"
    )

    charger_count = c2.number_input(
        "2. Previous Charger Sessions",
        min_value=0,
        value=10,
        step=1,
        key=key + "charger_count"
    )

    user_mean = c1.number_input(
        "3. Previous Average User Energy (kWh)",
        min_value=0.0,
        value=12.0,
        step=1.0,
        key=key + "user_mean"
    )

    hour = c2.number_input(
        "4. Charging Start Hour (0–23)",
        min_value=0,
        max_value=23,
        value=18,
        step=1,
        key=key + "hour"
    )

    user_count = c1.number_input(
        "5. Previous User Sessions",
        min_value=0,
        value=5,
        step=1,
        key=key + "user_count"
    )

    row = {
        "Charger_prev_mean": charger_mean,
        "Charger_prev_count": charger_count,
        "User_prev_mean": user_mean,
        "Hour": hour,
        "User_prev_count": user_count
    }

    X = pd.DataFrame([row]).reindex(
        columns=saved["columns"]
    )

    X = X.fillna(saved["fill_values"])

    return X


# ===================== CHART HELPER =====================
def bar(series, title):
    st.caption(title)
    st.bar_chart(series, color="#8ED9A1")


# ===================== SIDEBAR =====================
st.sidebar.title("⚡ EV CHARGING")
st.sidebar.caption("Smart Charging Insights")

# No NAVIGATION label
page = st.sidebar.radio(
    "",
    ["🏠 Home", "📈 Regression", "🏷️ Classification"],
    index=0,
    label_visibility="collapsed"
)


# ===================== HOME =====================
if page == "🏠 Home":

    st.title("⚡ EV Charging Dashboard")
    st.write(
        "Explore energy consumption, charger usage and charging patterns."
    )

    st.sidebar.divider()
    st.sidebar.subheader("🔎 Dashboard Filters")

    locs = st.sidebar.multiselect(
        "Location",
        sorted(df["Location"].dropna().unique())
    )

    types = st.sidebar.multiselect(
        "Charger Type",
        sorted(df["ChargerType"].dropna().unique())
    )

    comps = st.sidebar.multiselect(
        "Charger Company",
        sorted(df["ChargerCompany"].dropna().unique())
    )

    f = df.copy()

    if locs:
        f = f[f["Location"].isin(locs)]

    if types:
        f = f[f["ChargerType"].isin(types)]

    if comps:
        f = f[f["ChargerCompany"].isin(comps)]

    if f.empty:
        st.warning("No sessions match the selected filters.")
        st.stop()

    # KPI CARDS
    st.subheader("📊 Key Performance Indicators")

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Total Sessions",
        f"{len(f):,}"
    )

    k2.metric(
        "Total Energy",
        f"{f['Demand'].sum():,.0f} kWh"
    )

    k3.metric(
        "Average Energy",
        f"{f['Demand'].mean():.1f} kWh"
    )

    k4.metric(
        "Average Duration",
        f"{f['Duration'].mean():.0f} min"
    )

    k5.metric(
        "Unique Chargers",
        f"{f['ChargerID'].nunique():,}"
    )

    st.divider()
    st.subheader("🌱 Charging Demand Analysis")

    # CHART 1
    st.markdown("**1. Energy Demand by Location**")

    bar(
        f.groupby("Location")["Demand"].sum().sort_values(),
        "Compare total energy demand across locations."
    )

    # CHARTS 2 AND 3
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**2. Demand by Charger Type**")

        bar(
            f.groupby("ChargerType")["Demand"].sum(),
            "Energy consumption by charger type."
        )

    with c2:
        st.markdown("**3. Demand by Charger Company**")

        bar(
            f.groupby("ChargerCompany")["Demand"].sum(),
            "Energy consumption by company."
        )

    # CHARTS 4 AND 5
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**4. Demand by Start Hour**")

        bar(
            f.groupby("Hour")["Demand"].sum().sort_index(),
            "Identify peak charging hours."
        )

    with c4:
        st.markdown("**5. Demand by Day of Week**")

        weekly = (
            f.groupby("DayOfWeek")["Demand"]
            .sum()
            .reindex(range(7), fill_value=0)
        )

        weekly.index = DAYS

        bar(
            weekly,
            "Compare demand across the week."
        )


# ===================== REGRESSION =====================
elif page == "📈 Regression":

    st.title("📈 Energy Demand Prediction")

    st.write(
        "Predict energy consumption for a charging session."
    )

    X_new = session_form("reg_")

    if st.button(
        "🔮 Predict Energy Demand",
        key="reg_btn"
    ):
        try:
            prediction = saved["reg_model"].predict(X_new)[0]

            st.subheader("🌿 Prediction Result")

            st.metric(
                "Predicted Energy Demand",
                f"{max(0, float(prediction)):.2f} kWh"
            )

            st.success("Prediction completed!")

        except Exception as e:
            st.error(f"Regression prediction failed: {e}")


# ===================== CLASSIFICATION =====================
else:

    st.title("🏷️ Charging Demand Classification")

    st.write(
        "Predict whether a session has High or Normal demand."
    )

    # High demand threshold message removed
    X_new = session_form("clf_")

    if st.button(
        "🔮 Predict Demand Level",
        key="clf_btn"
    ):
        try:
            model = saved["clf_model"]

            probabilities = model.predict_proba(X_new)[0]
            classes = list(model.classes_)

            if 1 in classes:
                high_probability = probabilities[classes.index(1)]
            else:
                high_probability = probabilities[-1]

            prediction = (
                "High"
                if high_probability >= 0.5
                else "Normal"
            )

            st.subheader("🌿 Classification Result")

            c1, c2 = st.columns(2)

            c1.metric(
                "High Demand Probability",
                f"{high_probability:.1%}"
            )

            c2.metric(
                "Predicted Demand Level",
                prediction
            )

            if prediction == "High":
                st.warning(
                    "The model predicts High energy demand."
                )
            else:
                st.success(
                    "The model predicts Normal energy demand."
                )

        except Exception as e:
            st.error(f"Classification prediction failed: {e}")
