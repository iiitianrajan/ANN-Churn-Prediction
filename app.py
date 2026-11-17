"""Deep-learning ANN customer churn prediction dashboard.

Required files in the same directory:
    model.h5
    label_encoder_gender.pkl
    onehot_encoder_geo.pkl
    scaler.pkl

Run: streamlit run app.py
"""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Customer Churn | Deep Learning ANN",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: #f7f9fc; color: #17243b; }
        .block-container { max-width: 1120px; padding-top: 2.1rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8edf4; }
        [data-testid="stHeader"] { background: transparent; }
        h1, h2, h3 { letter-spacing: -0.035em; color: #13223a; }
        h1 { font-weight: 750; }
        div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
            border-color: #e4eaf3;
        }
        [data-testid="stForm"] {
            border: 1px solid #e5ebf4;
            border-radius: 16px;
            background: #ffffff;
            padding: 1.4rem 1.6rem 1rem;
            box-shadow: 0 4px 25px rgba(26, 49, 81, 0.035);
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6eaf1;
            border-radius: 12px;
            padding: 1rem 1.2rem;
        }
        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button[kind="primary"] {
            background: #2457d6;
            border-color: #2457d6;
            border-radius: 10px;
            font-weight: 650;
            padding: 0.65rem 1.2rem;
        }
        .eyebrow { color: #2457d6; font-size: 0.76rem; font-weight: 750;
                   letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.4rem; }
        .hero-text { color: #64748b; max-width: 690px; font-size: 1.02rem; line-height: 1.65; }
        .section-label { color: #66768e; font-weight: 700; font-size: 0.86rem;
                         letter-spacing: 0.015em; margin: 1rem 0 0.4rem; }
        .result-box { padding: 1.15rem 1.3rem; border-radius: 13px; margin: 0.8rem 0 1rem; }
        .result-high { background: #fff4f2; border: 1px solid #ffd9d4; }
        .result-low { background: #f0fbf6; border: 1px solid #c9efdc; }
        .result-heading { font-size: 1.14rem; font-weight: 750; margin-bottom: 0.25rem; }
        .result-copy { color: #475569; font-size: 0.94rem; }
        .footnote { color: #8190a6; font-size: 0.81rem; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_assets():
    root = Path(__file__).resolve().parent
    model = tf.keras.models.load_model(root / "model.h5", compile=False)
    with (root / "label_encoder_gender.pkl").open("rb") as f:
        gender_encoder = pickle.load(f)
    with (root / "onehot_encoder_geo.pkl").open("rb") as f:
        geo_encoder = pickle.load(f)
    with (root / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    return model, gender_encoder, geo_encoder, scaler


try:
    model, gender_encoder, geo_encoder, scaler = load_assets()
except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
    st.error("Unable to load model assets. Make sure the four trained files are next to app.py.")
    st.code("model.h5\nlabel_encoder_gender.pkl\nonehot_encoder_geo.pkl\nscaler.pkl")
    st.caption(f"Technical details: {exc}")
    st.stop()


with st.sidebar:
    st.markdown("### ◈ Churn Intelligence")
    st.caption("CUSTOMER RETENTION · DEEP LEARNING")
    st.divider()
    st.markdown("**About the project**")
    st.write(
        "An interactive deep-learning application powered by a trained "
        "Artificial Neural Network (ANN) that predicts bank-customer churn risk."
    )
    st.markdown("**Technology**")
    st.caption("Deep Learning (ANN) · TensorFlow / Keras · scikit-learn · Pandas · Streamlit")
    st.markdown("**Prediction logic**")
    st.caption("ANN binary classifier · 50% decision threshold")
    st.divider()
    st.info("Set customer details, then select **Predict churn risk** to view the estimate.", icon="💡")

st.markdown('<div class="eyebrow">Deep learning / artificial neural network (ANN)</div>', unsafe_allow_html=True)
st.title("Customer Churn Prediction with ANN")
st.markdown(
    '<p class="hero-text">Explore customer characteristics and estimate the likelihood '
    'of churn using a trained Artificial Neural Network (ANN).</p>',
    unsafe_allow_html=True,
)
with st.expander("About the ANN architecture"):
    st.write("This application performs **binary classification** using a saved TensorFlow/Keras ANN. ")
    st.caption("Details below come from the loaded model rather than assumed training settings.")
    try:
        st.metric("Trainable and non-trainable parameters", f"{model.count_params():,}")
        architecture = []
        for idx, layer in enumerate(model.layers, start=1):
            item = {
                "Layer": idx,
                "Type": layer.__class__.__name__,
                "Units": getattr(layer, "units", "—"),
                "Activation": getattr(getattr(layer, "activation", None), "__name__", "—"),
            }
            architecture.append(item)
        if architecture:
            st.dataframe(pd.DataFrame(architecture), hide_index=True, use_container_width=True)
    except Exception:
        st.caption("Architecture metadata is unavailable for this saved model.")
st.write("")

with st.form("customer_input_form"):
    st.subheader("Customer details")
    st.caption("Enter the information below to generate a prediction.")

    st.markdown('<div class="section-label">01 / Customer profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        geography = st.selectbox("Geography", options=list(geo_encoder.categories_[0]))
    with c2:
        gender = st.selectbox("Gender", options=list(gender_encoder.classes_))
    with c3:
        age = st.slider("Age", min_value=18, max_value=92, value=35)

    st.markdown('<div class="section-label">02 / Banking relationship</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)
    with c2:
        num_of_products = st.selectbox("Number of products", options=[1, 2, 3, 4])
    with c3:
        has_cr_card = st.selectbox("Has credit card?", options=[1, 0], format_func=lambda v: "Yes" if v else "No")
    is_active_member = st.selectbox(
        "Active member?", options=[1, 0], format_func=lambda v: "Yes" if v else "No"
    )

    st.markdown('<div class="section-label">03 / Financial information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        credit_score = st.number_input("Credit score", min_value=0, max_value=1000, value=650, step=1)
    with c2:
        balance = st.number_input("Account balance", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    with c3:
        estimated_salary = st.number_input("Estimated salary", min_value=0.0, value=60000.0, step=1000.0, format="%.2f")

    st.write("")
    submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

if submitted:
    try:
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [int(gender_encoder.transform([gender])[0])],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        })

        geo_encoded = geo_encoder.transform([[geography]])
        if hasattr(geo_encoded, "toarray"):
            geo_encoded = geo_encoded.toarray()
        geo_df = pd.DataFrame(
            geo_encoded,
            columns=geo_encoder.get_feature_names_out(["Geography"]),
        )
        input_data = pd.concat([input_data, geo_df], axis=1)

        # Match the feature order the scaler saw during training, when available.
        if hasattr(scaler, "feature_names_in_"):
            input_data = input_data.loc[:, list(scaler.feature_names_in_)]

        scaled = scaler.transform(input_data)
        probability = float(model.predict(scaled, verbose=0).reshape(-1)[0])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Model returned a value outside the 0–1 probability range.")

    except Exception as exc:
        st.error("The prediction could not be generated. Check the input and training pipeline.")
        with st.expander("Technical error"):
            st.exception(exc)
    else:
        st.write("")
        st.subheader("ANN prediction results")
        high_risk = probability >= 0.50
        style = "result-high" if high_risk else "result-low"
        heading = "Higher churn risk" if high_risk else "Lower churn risk"
        copy = (
            "The model estimates that this customer is more likely to leave than stay."
            if high_risk else
            "The model estimates that this customer is more likely to stay than leave."
        )
        st.markdown(
            f'<div class="result-box {style}">'
            f'<div class="result-heading">{heading}</div>'
            f'<div class="result-copy">{copy}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.metric("ANN-estimated churn probability", f"{probability:.1%}")
        with right:
            st.markdown("**Churn probability**")
            st.progress(probability)
            st.caption("Threshold for higher risk: 50%")

        with st.expander("View model input features"):
            st.dataframe(input_data, hide_index=True, use_container_width=True)

st.markdown(
    '<p class="footnote">Portfolio demo · Predictions are model estimates, not guarantees. '
    'No model accuracy or calibration claims are made here.</p>',
    unsafe_allow_html=True,
)
