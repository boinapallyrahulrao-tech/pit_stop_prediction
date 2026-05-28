import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go

# ==============================
# LOAD MODEL AND DATA
# ==============================

model = joblib.load("models/f1_pit_stop_final_pipeline.pkl")
train_data = pd.read_csv("data/raw/train.csv")

drivers = sorted(train_data["Driver"].unique())
compounds = sorted(train_data["Compound"].unique())
races = sorted(train_data["Race"].unique())

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="F1 Pit Stop Prediction",
    page_icon="🏎️",
    layout="wide"
)

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

header[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

h1 {
    color: #ff4b4b;
    text-align: center;
    font-size: 52px !important;
    font-weight: 800;
}

h2, h3 {
    color: #facc15;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

label {
    color: white !important;
    font-weight: 600;
}

.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff914d);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.result-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 22px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    margin-top: 20px;
}

p {
    color: #e5e7eb;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================

st.title("🏎️ F1 Pit Stop Prediction Dashboard")
st.write(
    "AI-powered Formula 1 pit stop prediction system using XGBoost, "
    "race strategy, tyre degradation, and telemetry features."
)

# ==============================
# SIDEBAR INPUTS
# ==============================

st.sidebar.header("Race Telemetry Inputs")

driver = st.sidebar.selectbox("Driver", drivers)
compound = st.sidebar.selectbox("Tyre Compound", compounds)
race = st.sidebar.selectbox("Race", races)

year = st.sidebar.number_input("Year", min_value=2018, max_value=2026, value=2023)
pitstop = st.sidebar.selectbox("Pit Stop This Lap?", [0, 1])
lap_number = st.sidebar.number_input("Lap Number", min_value=1, max_value=100, value=24)
stint = st.sidebar.number_input("Stint", min_value=1, max_value=10, value=2)
tyre_life = st.sidebar.number_input("Tyre Life", min_value=0.0, max_value=80.0, value=24.0)
position = st.sidebar.number_input("Current Position", min_value=1, max_value=20, value=8)
lap_time = st.sidebar.number_input("Lap Time (s)", min_value=0.0, value=90.0)
lap_time_delta = st.sidebar.number_input("Lap Time Delta", value=1.5)
cumulative_degradation = st.sidebar.number_input("Cumulative Degradation", value=5.0)
race_progress = st.sidebar.slider("Race Progress", min_value=0.0, max_value=1.0, value=0.45)
position_change = st.sidebar.number_input("Position Change", value=0.0)

# ==============================
# INPUT DATAFRAME
# ==============================

input_data = pd.DataFrame({
    "Driver": [driver],
    "Compound": [compound],
    "Race": [race],
    "Year": [year],
    "PitStop": [pitstop],
    "LapNumber": [lap_number],
    "Stint": [stint],
    "TyreLife": [tyre_life],
    "Position": [position],
    "LapTime (s)": [lap_time],
    "LapTime_Delta": [lap_time_delta],
    "Cumulative_Degradation": [cumulative_degradation],
    "RaceProgress": [race_progress],
    "Position_Change": [position_change]
})

# ==============================
# FEATURE ENGINEERING
# ==============================

def create_features(df):
    df = df.copy()

    df["TyreLife_Ratio"] = df["TyreLife"] / df["LapNumber"]
    df["PaceLoss_PerLap"] = df["LapTime_Delta"] / (df["TyreLife"] + 1)
    df["Deg_PerLap"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)

    df["PitWindow"] = (
        ((df["LapNumber"] >= 10) & (df["LapNumber"] <= 35)).astype(int)
    )

    def race_phase(progress):
        if progress < 0.33:
            return "Early"
        elif progress < 0.66:
            return "Mid"
        else:
            return "Late"

    df["RacePhase"] = df["RaceProgress"].apply(race_phase)
    df["Top10"] = (df["Position"] <= 10).astype(int)
    df["LosingPositions"] = (df["Position_Change"] < 0).astype(int)
    df["FreshTyres"] = (df["TyreLife"] <= 3).astype(int)
    df["LongStint"] = (df["TyreLife"] >= 20).astype(int)
    df["AggressiveDeg"] = (df["LapTime_Delta"] > 0).astype(int)

    return df


input_data = create_features(input_data)

# ==============================
# MAIN DISPLAY
# ==============================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Race Scenario")
    st.write(f"**Driver:** {driver}")
    st.write(f"**Compound:** {compound}")
    st.write(f"**Race:** {race}")
    st.write(f"**Lap Number:** {lap_number}")
    st.write(f"**Tyre Life:** {tyre_life}")

with col2:
    st.subheader("Model Details")
    st.write("**Model:** XGBoost")
    st.write("**Task:** Binary Classification")
    st.write("**Target:** PitNextLap")
    st.write("**Goal:** Predict next-lap pit stop")

st.divider()

# ==============================
# PREDICTION
# ==============================

if st.sidebar.button("Predict Pit Stop"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result_text = "PIT NEXT LAP"
        result_color = "#ef4444"
        result_message = "The model predicts that the driver is likely to pit on the next lap."
    else:
        result_text = "NO PIT NEXT LAP"
        result_color = "#22c55e"
        result_message = "The model predicts that the driver is unlikely to pit on the next lap."

    st.markdown(
        f"""
        <div class="result-card">
            <h2 style="color:{result_color}; margin-bottom:10px;">
                {result_text}
            </h2>
            <p style="font-size:18px;">
                {result_message}
            </p>
            <h3 style="color:white;">
                Pit Stop Probability: {probability:.2%}
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(probability))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Pit Stop Risk Meter"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ff4b4b"},
            "steps": [
                {"range": [0, 40], "color": "#16a34a"},
                {"range": [40, 70], "color": "#facc15"},
                {"range": [70, 100], "color": "#dc2626"}
            ],
        }
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tyre Wear Indicator")

    tyre_wear_score = min(tyre_life / 40, 1.0)
    st.progress(float(tyre_wear_score))

    if tyre_wear_score < 0.4:
        st.success("Low tyre wear")
    elif tyre_wear_score < 0.7:
        st.warning("Medium tyre wear")
    else:
        st.error("High tyre wear")

    st.subheader("Why This Prediction?")

    try:
        preprocessor = model.named_steps["preprocessor"]
        xgb_model = model.named_steps["model"]

        processed_input = preprocessor.transform(input_data)

        try:
            processed_input_dense = processed_input.toarray()
        except Exception:
            processed_input_dense = processed_input

        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(processed_input_dense)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Value": shap_values[0]
        })

        shap_df["Impact"] = shap_df["SHAP_Value"].apply(
            lambda x: "Increases pit probability" if x > 0 else "Decreases pit probability"
        )

        shap_df["Abs_SHAP"] = shap_df["SHAP_Value"].abs()

        top_shap = shap_df.sort_values(
            by="Abs_SHAP",
            ascending=False
        ).head(10)

        st.dataframe(top_shap[["Feature", "SHAP_Value", "Impact"]])

    except Exception as error:
        st.warning("SHAP explanation could not be generated.")
        st.write(error)

    st.subheader("Input Data")
    st.dataframe(input_data)

else:
    st.info("Enter race details in sidebar and click Predict Pit Stop.")