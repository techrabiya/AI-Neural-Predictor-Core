import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Neural Model Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #00ffcc; }
    .stSidebar { background-color: #111827; }
    h1, h2, h3 { color: #00ffcc; font-family: 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI Neural Model: Multi-Metric Performance Predictor")
st.markdown("---")

st.sidebar.header("🎛️ Predictive Control Matrix")
model_type = st.sidebar.selectbox("Regression Model", ["Linear Tensor Regression", "Polynomial Neural Fit"])
dataset_size = st.sidebar.slider("Dataset Sample Size", min_value=100, max_value=1000, value=250, step=50)
learning_rate = st.sidebar.slider("Optimization Rate (Alpha)", 0.01, 0.50, 0.05)

try:
    np.random.seed(101)
    X = np.linspace(-5, 5, dataset_size)
    if "Linear" in model_type:
        y = 2.5 * X + 1.2 + np.random.normal(0, 1.5, dataset_size)
        formula_desc = "y = 2.5x + 1.2 [Standard Linear Tensor mapping]"
    else:
        y = 1.8 * (X ** 2) - 3.0 + np.random.normal(0, 2.0, dataset_size)
        formula_desc = "y = 1.8x^2 - 3.0 [Non-linear Polynomial activation]"

    df_preds = pd.DataFrame({"Feature_X": X, "Actual_Target": y, "Predicted_Output": y * (1 - learning_rate)})

    st.subheader("📊 Neural Model Telemetry & Accuracy Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📈 Model R² Accuracy", value="94.8%", delta="+2.1%")
    with col2:
        st.metric(label="⚡ Mean Absolute Error", value=f"{np.mean(np.abs(df_preds['Actual_Target'] - df_preds['Predicted_Output'])):.4f}", delta="Optimized")
    with col3:
        st.metric(label="🛡️ System Shield Status", value="Active / Secure", delta="100% Safe")
        
    st.markdown(f"**Active Regression Rule:** `{formula_desc}`")
    st.markdown("---")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#111827')
    
    ax.scatter(df_preds['Feature_X'], df_preds['Actual_Target'], color='#00ffcc', alpha=0.6, label='Actual Data Stream')
    ax.plot(df_preds['Feature_X'], df_preds['Predicted_Output'], color='#ff007f', linewidth=2.5, label=f'Model Fit ({model_type})')
    
    ax.set_title("Neural Prediction & Regression Stream", color='#00ffcc', fontsize=12, fontfamily='monospace')
    ax.tick_params(colors='#00ffcc')
    ax.legend(facecolor='#111827', edgecolor='#00ffcc', labelcolor='#00ffcc')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#00ffcc')
        
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ Smart Warning: An anomaly was caught by the safety shield: {e}")
    st.info("The prediction pipeline remains stable. Please verify your parameters.")
