import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD TRAINED ML COMPONENTS
# =========================================================

model = joblib.load(
    "breast_cancer_random_forest_model.pkl"
)

scaler = joblib.load(
    "breast_cancer_scaler.pkl"
)

selector = joblib.load(
    "breast_cancer_feature_selector.pkl"
)


# =========================================================
# TITLE
# =========================================================

st.title("🩺 Breast Cancer Prediction System")

st.write(
    "Machine Learning-Based Breast Cancer Classification"
)

st.warning(
    "Research prototype using synthetic data. "
    "This system is not a medical diagnostic tool."
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.header("1. Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=27.5
    )


# =========================================================
# VITAL & LABORATORY FEATURES
# =========================================================

st.header("2. Vital & Laboratory Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    systolic_bp = st.number_input(
        "Systolic BP (mmHg)",
        value=130
    )

with col2:
    diastolic_bp = st.number_input(
        "Diastolic BP (mmHg)",
        value=82
    )

with col3:
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        value=210
    )

with col4:
    glucose = st.number_input(
        "Glucose (mg/dL)",
        value=100
    )


# =========================================================
# MEDICAL HISTORY
# =========================================================

st.header("3. Medical History")

col1, col2, col3 = st.columns(3)

with col1:
    family_history = st.selectbox(
        "Family History of Breast Cancer",
        ["No", "Yes"]
    )

with col2:
    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

with col3:
    alcohol_use = st.selectbox(
        "Alcohol Use",
        ["No", "Yes"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    menopause = st.selectbox(
        "Menopause",
        ["No", "Yes"]
    )

with col2:
    hormonal_therapy = st.selectbox(
        "Hormonal Therapy",
        ["No", "Yes"]
    )

with col3:
    previous_benign = st.selectbox(
        "Previous Benign Breast Disease",
        ["No", "Yes"]
    )


# =========================================================
# BREAST CLINICAL FEATURES
# =========================================================

st.header("4. Breast Clinical Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    breast_lump = st.selectbox(
        "Breast Lump",
        ["No", "Yes"]
    )

with col2:
    breast_pain = st.selectbox(
        "Breast Pain",
        ["No", "Yes"]
    )

with col3:
    nipple_discharge = st.selectbox(
        "Nipple Discharge",
        ["No", "Yes"]
    )

with col4:
    skin_changes = st.selectbox(
        "Skin Changes",
        ["No", "Yes"]
    )


# =========================================================
# TUMOR / IMAGING FEATURES
# =========================================================

st.header("5. Tumor & Imaging Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    axillary_nodes = st.number_input(
        "Axillary Lymph Nodes",
        min_value=0.0,
        value=2.0
    )

with col2:
    tumor_size = st.number_input(
        "Tumor Size (mm)",
        min_value=0.0,
        value=24.0
    )

with col3:
    birads = st.number_input(
        "BI-RADS Score",
        min_value=1,
        max_value=5,
        value=4
    )

with col4:
    ultrasound = st.number_input(
        "Ultrasound Irregularity",
        min_value=0.0,
        max_value=1.0,
        value=0.75
    )

mammographic_density = st.number_input(
    "Mammographic Density Score",
    min_value=0.0,
    max_value=1.0,
    value=0.50
)


# =========================================================
# NUCLEUS FEATURES
# =========================================================

st.header("6. Tumor Cell / Nucleus Features")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    radius = st.number_input(
        "Nucleus Radius",
        value=17.0
    )

with col2:
    perimeter = st.number_input(
        "Nucleus Perimeter",
        value=110.0
    )

with col3:
    area = st.number_input(
        "Nucleus Area",
        value=700.0
    )

with col4:
    texture = st.number_input(
        "Nucleus Texture",
        value=23.0
    )

with col5:
    smoothness = st.number_input(
        "Nucleus Smoothness",
        value=0.105
    )

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    compactness = st.number_input(
        "Nucleus Compactness",
        value=0.18
    )

with col2:
    concavity = st.number_input(
        "Nucleus Concavity",
        value=0.15
    )

with col3:
    concave_points = st.number_input(
        "Nucleus Concave Points",
        value=0.08
    )

with col4:
    symmetry = st.number_input(
        "Nucleus Symmetry",
        value=0.20
    )

with col5:
    fractal_dimension = st.number_input(
        "Nucleus Fractal Dimension",
        value=0.064
    )


# =========================================================
# CONVERT YES/NO TO 0/1
# =========================================================

def yes_no(value):
    return 1 if value == "Yes" else 0


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 PREDICT",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    patient = pd.DataFrame({

        "Age_years": [age],

        "Gender": [
            0 if gender == "Female" else 1
        ],

        "Systolic_BP_mmHg": [systolic_bp],
        "Diastolic_BP_mmHg": [diastolic_bp],
        "Cholesterol_mg_dL": [cholesterol],
        "Glucose_mg_dL": [glucose],
        "BMI_kg_m2": [bmi],

        "Family_History_Breast_Cancer":
            [yes_no(family_history)],

        "Smoking":
            [yes_no(smoking)],

        "Alcohol_Use":
            [yes_no(alcohol_use)],

        "Menopause":
            [yes_no(menopause)],

        "Hormonal_Therapy":
            [yes_no(hormonal_therapy)],

        "Previous_Benign_Breast_Disease":
            [yes_no(previous_benign)],

        "Breast_Lump":
            [yes_no(breast_lump)],

        "Breast_Pain":
            [yes_no(breast_pain)],

        "Nipple_Discharge":
            [yes_no(nipple_discharge)],

        "Skin_Changes":
            [yes_no(skin_changes)],

        "Axillary_Lymph_Nodes":
            [axillary_nodes],

        "Tumor_Size_mm":
            [tumor_size],

        "BI_RADS":
            [birads],

        "Ultrasound_Irregularity_Score":
            [ultrasound],

        "Mammographic_Density_Score":
            [mammographic_density],

        "Nucleus_Radius":
            [radius],

        "Nucleus_Perimeter":
            [perimeter],

        "Nucleus_Area":
            [area],

        "Nucleus_Texture":
            [texture],

        "Nucleus_Smoothness":
            [smoothness],

        "Nucleus_Compactness":
            [compactness],

        "Nucleus_Concavity":
            [concavity],

        "Nucleus_Concave_Points":
            [concave_points],

        "Nucleus_Symmetry":
            [symmetry],

        "Nucleus_Fractal_Dimension":
            [fractal_dimension]
    })


    # =====================================================
    # SCALE
    # =====================================================

    patient_scaled = scaler.transform(patient)


    # =====================================================
    # FEATURE SELECTION
    # =====================================================

    patient_selected = selector.transform(
        patient_scaled
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(
        patient_selected
    )

    probability = model.predict_proba(
        patient_selected
    )[0][1]


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.header("7. Prediction Result")

    if prediction[0] == 1:

        st.error(
            "Prediction: MALIGNANT"
        )

    else:

        st.success(
            "Prediction: BENIGN"
        )

    st.metric(
        "Model Score",
        f"{probability * 100:.2f}%"
    )

    st.info(
        "This result is generated by the trained machine-learning "
        "model and should not be interpreted as a clinical diagnosis."
    )