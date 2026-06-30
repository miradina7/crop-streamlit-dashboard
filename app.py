import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

# --- Page Configurations ---
st.set_page_config(
    page_title="Crop Recommendation Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Custom Styling (Avoids generic overrides to prevent layout bugs or blurry elements) ---
st.markdown("""
    <style>
    /* Clean header container */
    .header-container {
        background: linear-gradient(135deg, #e0f2f1 0%, #e8f5e9 100%);
        padding: 20px 15px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 15px;
        border-left: 8px solid #00796b;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .header-title {
        color: #004d40 !important;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 4px;
        margin-top: 0;
    }
    .header-subtitle {
        color: #2e7d32 !important;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 0;
    }
    
    /* Result card highlight style */
    .result-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #b2dfdb 100%);
        border: 2px solid #80cbc4;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0, 77, 64, 0.08);
        margin-top: 15px;
        color: #004d40;
    }
    .result-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #00796b;
        margin-bottom: 5px;
    }
    .result-crop {
        font-size: 2.3rem;
        font-weight: 900;
        color: #004d40;
        text-transform: uppercase;
        margin: 5px 0;
        letter-spacing: 1.5px;
    }
    .result-desc {
        color: #1b5e20;
        font-size: 0.9rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Fix sidebar dropdown and input readability */
    [data-testid="stSidebar"] * {
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #111827 !important;
    }

    div[data-baseweb="select"] * {
        color: #111827 !important;
        opacity: 1 !important;
    }

    div[data-baseweb="popover"] * {
        color: #111827 !important;
        opacity: 1 !important;
    }

    input {
        color: #111827 !important;
        opacity: 1 !important;
    }

    </style>
""", unsafe_allow_html=True)


# --- Fallback Dataset Generator ---
def create_fallback_dataset():
    """Generates a synthetic agricultural dataset matching standard columns if Crop_recommendation.csv is missing."""
    import numpy as np
    np.random.seed(42)
    crops = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
        'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes',
        'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
    ]
    data = []
    for crop in crops:
        for _ in range(35):
            data.append({
                'N': np.random.randint(10, 140),
                'P': np.random.randint(10, 100),
                'K': np.random.randint(10, 205),
                'temperature': np.random.uniform(15.0, 38.0),
                'humidity': np.random.uniform(35.0, 95.0),
                'ph': np.random.uniform(4.5, 8.5),
                'rainfall': np.random.uniform(40.0, 300.0),
                'label': crop
            })
    return pd.DataFrame(data)


# --- Data Loader ---
@st.cache_data
def load_data():
    file_path = "Crop_recommendation.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path), False
    else:
        return create_fallback_dataset(), True

df, is_mock = load_data()


# --- Machine Learning Pipeline ---
@st.cache_resource
def train_model(dataframe):
    X = dataframe[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = dataframe['label']
    
    # 80% train and 20% test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standard DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    classes_count = len(y.unique())
    
    return model, acc, classes_count

trained_model, model_acc, total_classes = train_model(df)


# --- Dynamic Setup of Presets ---
preset_to_label = {
    "Rice Example": "rice",
    "Maize Example": "maize",
    "Chickpea Example": "chickpea",
    "Kidneybeans Example": "kidneybeans",
    "Pigeonpeas Example": "pigeonpeas",
    "Mothbeans Example": "mothbeans",
    "Mungbean Example": "mungbean",
    "Blackgram Example": "blackgram",
    "Lentil Example": "lentil",
    "Pomegranate Example": "pomegranate",
    "Banana Example": "banana",
    "Mango Example": "mango",
    "Grapes Example": "grapes",
    "Watermelon Example": "watermelon",
    "Muskmelon Example": "muskmelon",
    "Apple Example": "apple",
    "Orange Example": "orange",
    "Papaya Example": "papaya",
    "Coconut Example": "coconut",
    "Cotton Example": "cotton",
    "Jute Example": "jute",
    "Coffee Example": "coffee"
}

@st.cache_data
def calculate_preset_means(dataframe):
    df_copy = dataframe.copy()
    df_copy['label_clean'] = df_copy['label'].astype(str).str.strip().str.lower()
    means = df_copy.groupby('label_clean')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean().to_dict(orient='index')
    return means

preset_means_dict = calculate_preset_means(df)

# Assemble options
preset_means = {
    "Custom Input": {
        "N": 50.0, "P": 50.0, "K": 50.0, "temperature": 25.0, "humidity": 70.0, "ph": 6.5, "rainfall": 100.0
    }
}

for preset_name, label_key in preset_to_label.items():
    if label_key in preset_means_dict:
        preset_means[preset_name] = preset_means_dict[label_key]
    else:
        preset_means[preset_name] = {
            "N": 50.0, "P": 50.0, "K": 50.0, "temperature": 25.0, "humidity": 70.0, "ph": 6.5, "rainfall": 100.0
        }


# --- Interactive Presets State Sync Callback ---
def sync_preset_state():
    selected = st.session_state.selected_preset
    vals = preset_means[selected]
    st.session_state.n_val = float(vals["N"])
    st.session_state.p_val = float(vals["P"])
    st.session_state.k_val = float(vals["K"])
    st.session_state.temp_val = float(vals["temperature"])
    st.session_state.humidity_val = float(vals["humidity"])
    st.session_state.ph_val = float(vals["ph"])
    st.session_state.rainfall_val = float(vals["rainfall"])

if "n_val" not in st.session_state:
    st.session_state.selected_preset = "Custom Input"
    sync_preset_state()


# --- Header Section ---
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🌾 Crop Recommendation Dashboard</h1>
        <p class="header-subtitle">Recommending suitable crops based on soil and climate parameters.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if is_mock:
    st.warning("⚠️ 'Crop_recommendation.csv' not found. Using dynamically generated synthetic agricultural data.")


# --- Outlined Model Profile Cards ---
def render_outlined_card(col, label, value):
    with col:
        st.markdown(
            f"""
            <div style="
                border: 1px solid #e0f2f1; 
                border-radius: 12px; 
                padding: 15px; 
                background-color: #ffffff; 
                text-align: center; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.03);
                min-height: 90px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <div style="
                    font-size: 0.75rem; 
                    color: #78909c; 
                    font-weight: 600; 
                    text-transform: uppercase; 
                    letter-spacing: 0.8px; 
                    margin-bottom: 6px;
                ">
                    {label}
                </div>
                <div style="
                    font-size: 1.1rem; 
                    color: #00796b; 
                    font-weight: 700; 
                    line-height: 1.2;
                ">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
render_outlined_card(col_m1, "Model Profile", "Decision Tree")
render_outlined_card(col_m2, "Machine Learning Task", "Multi-class Classification")
render_outlined_card(col_m3, "Classification Accuracy", f"{model_acc * 100:.2f}%")
render_outlined_card(col_m4, "Data Split Ratio", "80:20 (Train:Test)")
render_outlined_card(col_m5, "Total Crop Classes", f"{total_classes}")


# --- Sidebar Inputs & Configuration Parameters ---
st.sidebar.markdown("### 🛠️ Input Parameters")
st.sidebar.selectbox(
    "Choose Preset Example Scenario:",
    options=list(preset_means.keys()),
    key="selected_preset",
    on_change=sync_preset_state
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Soil Properties")
n_input = st.sidebar.number_input("Nitrogen (N) - mg/kg", min_value=0.0, max_value=250.0, step=1.0, key="n_val")
p_input = st.sidebar.number_input("Phosphorus (P) - mg/kg", min_value=0.0, max_value=250.0, step=1.0, key="p_val")
k_input = st.sidebar.number_input("Potassium (K) - mg/kg", min_value=0.0, max_value=300.0, step=1.0, key="k_val")

st.sidebar.markdown("#### Climate & Environmental Parameters")
temp_input = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, step=0.1, key="temp_val")
humidity_input = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, key="humidity_val")
ph_input = st.sidebar.number_input("Soil pH level", min_value=0.0, max_value=14.0, step=0.1, key="ph_val")
rainfall_input = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=450.0, step=1.0, key="rainfall_val")


# --- Main Dashboard Sections (Directly below Model Profile Cards, No empty space) ---
col_left, col_right = st.columns([1, 1])

# Left Column - Displays current inputs
with col_left:
    st.markdown("### 📋 Current Input Parameters")
    st.write("Current physical soil and climate parameters loaded for recommendation analysis:")
    
    # Summary table of inputs
    summary_df = pd.DataFrame({
        "Feature Metric": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "Soil pH", "Rainfall"],
        "Active Parameter Value": [
            f"{n_input:.1f} mg/kg", 
            f"{p_input:.1f} mg/kg", 
            f"{k_input:.1f} mg/kg", 
            f"{temp_input:.1f} °C", 
            f"{humidity_input:.1f} %", 
            f"{ph_input:.2f}", 
            f"{rainfall_input:.1f} mm"
        ]
    })
    st.table(summary_df)

# Right Column - Handles prediction processes and outputs
with col_right:
    st.markdown("### 🔮 Prediction Execution")
    st.write("Click the button below to analyze settings and predict the most suitable crop class:")
    
    predict_btn = st.button("🌱 Predict Suitable Crop", type="primary", use_container_width=True)
    
    if predict_btn:
        input_features = [[n_input, p_input, k_input, temp_input, humidity_input, ph_input, rainfall_input]]
        prediction = trained_model.predict(input_features)[0]
        
        # Display Outcome Card
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Recommended Optimal Crop</div>
                <div class="result-crop">{prediction}</div>
                <p class="result-desc">
                    These micro-climatic and soil parameters align closely with the growth requirements of {prediction}.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("ℹ️ Click the button to generate a recommendation.")


# --- Dataset Diagnostics Expandable Section ---
st.markdown("---")
with st.expander("📊 Dataset Diagnostics", expanded=False):
    tab_overview, tab_distribution = st.tabs(["🔬 Dataset Overview", "📈 Class Distribution"])

    with tab_overview:
        st.markdown("#### Primary Dataset Specifications")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.write(f"**Total Records (Dataset Size):** {df.shape[0]} samples")
            st.write("**Number of Input Features:** 7 features")
            st.write("**Output Target Variable:** `label` (crop label)")
        with col_o2:
            st.write(f"**Number of Crop Classes:** {total_classes} distinct crops")
            st.write("**Model Splitting Ratio:** 80% Training Data / 20% Evaluation Test set")
            st.write("**Replicability State:** random_state=42")

    with tab_distribution:
        st.markdown("#### Sample Frequencies per Crop Class")
        st.write("The chart shows the number of samples for each crop class. A balanced distribution helps reduce model bias.")
        
        # Calculate class count frequencies
        frequencies = df['label'].value_counts().reset_index()
        frequencies.columns = ['Crop', 'Samples Count']
        
        st.bar_chart(frequencies.set_index('Crop'), color="#00796b")