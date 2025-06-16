import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
#Firebase configuration file import
from firebase_config import db
from datetime import datetime

def log_user_input_to_firestore(inputs, prediction):
    db.collection("breast_cancer_predictions").add({
        "timestamp": datetime.utcnow(),
        "inputs": inputs,
        "prediction": prediction
    })

# Set Streamlit page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🧬",
    layout="wide"
)

# Load trained model
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load built-in dataset
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target  # 0 = malignant, 1 = benign
    return df

df = load_data()
feature_names = df.drop('diagnosis', axis=1).columns.tolist()

# Scale dataset for distribution plots
df_scaled = df.drop('diagnosis', axis=1)
robust_scaled = RobustScaler().fit_transform(df_scaled)
df_scaled = pd.DataFrame(robust_scaled, columns=df_scaled.columns)

X = df.drop('diagnosis', axis = 1)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components = 0.95)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns = [f'PCA{i}' for i in range(1, 11)])

# --- Plotly Figures ---

# Target Distribution
fig_dist = px.histogram(df, x='diagnosis', color='diagnosis',
                        color_discrete_sequence=['crimson', 'green'],
                        labels={'diagnosis': 'Diagnosis'},
                        title='Diagnosis Class Distribution')
fig_dist.update_xaxes(ticktext=["Malignant", "Benign"], tickvals=[0, 1])

# Correlation Heatmap
corr = df.corr()
fig_corr = ff.create_annotated_heatmap(
    z=corr.values.round(2),
    x=list(corr.columns),
    y=list(corr.index),
    colorscale='Viridis'
)
fig_corr.update_layout(
    width=1400,
    height=1000,
    title = ' ',
    title_x=0.5,
    margin=dict(l=80, r=80, t=80, b=80),
)

# Boxplots (30 features)
fig_boxplots = []
for feature in df.columns[:-1]:
    fig = px.box(df, x='diagnosis', y=feature, color='diagnosis',
                 color_discrete_sequence=['crimson', 'green'],
                 labels={'diagnosis': 'Diagnosis'},
                 title=f'Boxplot of {feature} by Diagnosis')
    fig.update_xaxes(ticktext=["Malignant", "Benign"], tickvals=[0, 1])
    fig_boxplots.append((feature, fig))

# Feature Histograms (30 features)
fig_histograms = []
for feature in df_scaled.columns:
    fig = px.histogram(df_scaled, x=feature, nbins=40, marginal='box',
                       title=f'Distribution of {feature} (Robust Scaled)')
    fig_histograms.append((feature, fig))

# --- Streamlit App ---

# App Title
st.markdown("""
    <h2 style='text-align: center; color: #4B8BBE;'>🧬 Breast Cancer Detection App</h2>
    <p style='text-align: center;'>Use this tool to predict breast cancer and explore key diagnostic insights.</p>
    """, unsafe_allow_html=True)

# Sidebar for EDA
st.sidebar.markdown(
    "A Streamlit-based web app that uses an SVM model to predict whether a breast tumor is benign or malignant. It features real-time predictions, PCA-based visualizations, and showcases the role of AI in medical diagnostics."
)

st.sidebar.title("📊 Data Insights")
eda_option = st.sidebar.selectbox(
    "Choose an insight:",
    (
        "Prediction Software",
        "Target Distribution",
        "Correlation Heatmap",
        "Target Boxplot",
        "Feature Distribution",
        "PCA Plot",
        "Feature Boxplots"
    )
)

# EDA Visualization
if eda_option == "Target Distribution":
    st.subheader("Diagnosis Class Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

elif eda_option == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

elif eda_option == "Target Boxplot":
    st.subheader("Boxplots of Features by Diagnosis")
    for feature, fig in fig_boxplots:
        with st.expander(f"{feature}"):
            st.plotly_chart(fig, use_container_width=True)

elif eda_option == "Feature Distribution":
    st.subheader("Distribution of Each Scaled Feature")
    for feature, fig in fig_histograms:
        with st.expander(f"{feature}"):
            st.plotly_chart(fig, use_container_width=True)

elif eda_option == 'PCA Plot':
    colors = {0: 'orange', 1: 'green'}
    diagnosis_map = {0: 'Malignant', 1: 'Benign'}

    # Limit to first few principal components (e.g. first 6)
    pca_features = X_pca.columns[:6]

    index = 0
    max_plots = 12  # control total number of pairwise plots

    for i, a in enumerate(pca_features):
        for j, b in enumerate(pca_features):
            if a == b or index >= max_plots:
                continue

            fig = px.scatter(
                X_pca,
                x=a,
                y=b,
                color=df['diagnosis'].map(diagnosis_map),
                trendline="ols",
                color_discrete_map={"Malignant": "orange", "Benign": "green"},
                opacity=0.5,
                title=f"{a} vs {b} by Diagnosis"
            )

            with st.expander(f"{a} vs {b}"):
                st.plotly_chart(fig, use_container_width=True)

            index += 1

elif eda_option == 'Feature Boxplots':
    for feature in df_scaled.columns:
        fig = px.box(df_scaled, y=feature, points='all', title=f'Boxplot of {feature}')
    
        with st.expander(f"Boxplot — {feature}"):
            st.plotly_chart(fig, use_container_width=True)

# --- Prediction Section ---
elif eda_option == 'Prediction Software':
    st.markdown("---")
    st.markdown("### 🔢 Enter Feature Values for Prediction")

    input_data = []
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(f"{feature}", min_value=0.0, step=0.01, format="%.2f")
            input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        label = "Malignant" if prediction == 0 else "Benign"

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_array)[0]
            confidence = proba[prediction] * 100
            st.info(f"Model Confidence: {confidence:.2f}%")

        st.success(f"🩺 Prediction: **{label}**")

        input_dict = dict(zip(feature_names, input_array.flatten()))
        db.collection("breast_cancer_predictions").add({
        "timestamp": datetime.utcnow(),
            "user_inputs": input_dict,
            "prediction": label,
            "confidence": round(confidence, 2) if 'confidence' in locals() else None
        })
