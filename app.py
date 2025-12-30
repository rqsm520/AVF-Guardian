import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import itertools
import altair as alt

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AVF Guardian | Smart Risk Assessment",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    /* Global Style */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Styling */
    .main-title {
        color: #2c3e50;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0rem !important;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        color: #7f8c8d;
        font-size: 1.2rem !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 400;
    }
    
    /* Card Styling */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 5px solid #3498db;
    }
    
    /* Risk Badges */
    .risk-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        min-width: 120px;
    }
    .risk-low { background-color: #2ecc71; }
    .risk-moderate { background-color: #f1c40f; color: #2c3e50; }
    .risk-high { background-color: #e74c3c; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e6e6e6;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #333333 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1e3c72 !important;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    # Try different paths for robustness
    paths = [
        r"e:\MLR\Models",
        "Models",
        "../Models"
    ]
    base_path = None
    for p in paths:
        if os.path.exists(p):
            base_path = p
            break
            
    if not base_path:
        st.error("Model files not found! Please ensure 'Models' directory exists.")
        st.stop()
        
    try:
        model = joblib.load(os.path.join(base_path, "lr_model.pkl"))
        scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
        winsor_limits = joblib.load(os.path.join(base_path, "winsor_limits.pkl"))
        # Load stats if available, else use hardcoded defaults
        try:
            stats = joblib.load(os.path.join(base_path, "data_stats.pkl"))
        except:
            stats = {}
        return model, scaler, winsor_limits, stats
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

model, scaler, winsor_limits, stats = load_models()

# ---------------------------------------------------------
# 3. Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.title("Patient Parameters")
st.sidebar.markdown("Enter the 6 core variables:")

def get_default(col, fallback):
    if col in stats and '50%' in stats[col]:
        return float(stats[col]['50%'])
    return fallback

with st.sidebar.form("prediction_form"):
    # Group 1: Demographics & History
    st.markdown("### 1. Clinical History")
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female", help="Male=1, Female=2")
    ijvc = st.selectbox("History of IJV Cannulation", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No", help="Internal Jugular Vein Cannulation History")
    
    # Group 2: Inflammatory Markers
    st.markdown("### 2. Biomarkers")
    
    col1, col2 = st.columns(2)
    with col1:
        mlr = st.number_input("MLR", min_value=0.0, max_value=10.0, value=get_default('MLR', 0.4), step=0.01, help="Monocyte-to-Lymphocyte Ratio")
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=200.0, value=get_default('CRP', 5.0), step=0.1)
    
    with col2:
        nlr = st.number_input("NLR", min_value=0.0, max_value=50.0, value=get_default('NLR', 3.0), step=0.1, help="Neutrophil-to-Lymphocyte Ratio")
        tg = st.number_input("Triglycerides (mmol/L)", min_value=0.0, max_value=20.0, value=get_default('triglycerides', 1.5), step=0.1)

    submitted = st.form_submit_button("Predict Risk 🚀")

# ---------------------------------------------------------
# 4. Main Prediction Logic
# ---------------------------------------------------------
# Header Section
st.markdown('<h1 class="main-title">AVF Guardian</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Smart Risk Assessment for Arteriovenous Fistula Dysfunction</p>', unsafe_allow_html=True)

# Intro Card
st.markdown("""
<div class="card">
    <h4>👋 Welcome to the Clinical Decision Support System</h4>
    <p>This tool utilizes the <strong>Extreme Minimalist Machine Learning Model</strong> (validated on 726 patients) 
    to predict the risk of AVF dysfunction. It focuses on 6 key indicators, including novel inflammatory biomarkers 
    (MLR, NLR) and lipid metabolism metrics.</p>
</div>
""", unsafe_allow_html=True)

if submitted:
    # 4.1 Preprocessing
    input_data = {
        'MLR': mlr, 'CRP': crp, 'triglycerides': tg, 'NLR': nlr,
        'IJVC': ijvc, 'sex': sex
    }
    df_input = pd.DataFrame([input_data])
    
    # Winsorize & Log (Mirror training logic)
    numeric_cols = ['MLR', 'CRP', 'triglycerides', 'NLR']
    for col in numeric_cols:
        # Find limits (handle case sensitivity if needed)
        limits = None
        for k in winsor_limits:
            if k.strip().lower() == col.strip().lower():
                limits = winsor_limits[k]
                break
        
        val = df_input[col].values[0]
        if limits:
            val = max(limits['lower'], min(val, limits['upper']))
        
        # Log transform
        df_input[f'log_{col}'] = np.log1p(val)
        
    # Generate Features (Interactions)
    core_feats = ['log_MLR', 'log_CRP', 'log_triglycerides', 'log_NLR', 'IJVC', 'sex']
    
    # Ensure columns exist (IJVC and sex are already there)
    
    X_final = pd.DataFrame(index=[0])
    # Add Main Effects
    for f in core_feats:
        X_final[f] = df_input[f]
        
    # Add Interactions
    for c1, c2 in itertools.combinations(core_feats, 2):
        X_final[f"{c1}*{c2}"] = df_input[c1] * df_input[c2]
        
    # Scale
    try:
        X_scaled = scaler.transform(X_final)
        
        # Predict
        prob = model.predict_proba(X_scaled)[0, 1]
        
        # 4.2 Display Results
        st.divider()
        st.subheader("📊 Assessment Results")
        
        col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
        
        with col_res1:
            st.metric(label="Risk Probability", value=f"{prob*100:.1f}%")
        
        with col_res2:
            # Risk Level Badge
            if prob < 0.2:
                st.markdown('<div style="margin-top:20px"><span class="risk-badge risk-low">LOW RISK</span></div>', unsafe_allow_html=True)
            elif prob < 0.5:
                st.markdown('<div style="margin-top:20px"><span class="risk-badge risk-moderate">MODERATE RISK</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="margin-top:20px"><span class="risk-badge risk-high">HIGH RISK</span></div>', unsafe_allow_html=True)

        with col_res3:
             # Gauge Chart using Altair
            st.write("**Risk Scale**")
            
            gauge_chart = alt.Chart(pd.DataFrame({'value': [prob]})).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("value", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("value", scale=alt.Scale(domain=[0, 0.5, 1], range=['green', 'yellow', 'red']), legend=None),
                tooltip=['value']
            ).properties(width=200, height=200)
            
            text = alt.Chart(pd.DataFrame({'value': [f"{prob:.1%}"]})).mark_text(align='center', baseline='middle', fontSize=20, fontWeight='bold').encode(
                text='value'
            )
            
            st.altair_chart(gauge_chart + text, use_container_width=True)
            
            # Suggestion Box
            st.markdown("### 📋 Clinical Interpretation & Recommendations")
            
            if prob > 0.5:
                st.error("""
                **🔴 High Risk Category**
                
                **Interpretation**: This patient has a significantly elevated risk of AVF dysfunction. The model estimates a >50% probability of failure or significant stenosis.
                
                **Recommended Actions**:
                - **Immediate**: Refer for Duplex Ultrasound surveillance to assess flow volume and anatomy.
                - **Monitoring**: Increase frequency of physical examination (e.g., every dialysis session) and dynamic venous pressure monitoring.
                - **Intervention**: Review modifiable risk factors, specifically inflammation (CRP, MLR/NLR optimization) and lipid management.
                """)
            elif prob > 0.2:
                st.warning("""
                **🟡 Moderate Risk Category**
                
                **Interpretation**: This patient shows intermediate signs of risk. While not critical, early warning signs may be present.
                
                **Recommended Actions**:
                - **Surveillance**: Monthly access flow monitoring (transonic or dilution) is recommended.
                - **Maintenance**: Ensure proper needle rotation and cannulation technique to preserve vessel integrity.
                - **Follow-up**: Re-assess risk markers (CRP, lipids) in 3 months.
                """)
            else:
                st.success("""
                **🟢 Low Risk Category**
                
                **Interpretation**: The estimated risk is within the baseline range for the dialysis population.
                
                **Recommended Actions**:
                - **Standard of Care**: Routine physical examination and standard surveillance protocols.
                - **Prevention**: Continue current maintenance therapy and lifestyle management.
                """)

        # 4.3 Interpretation (Contribution)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card"><h4>🔍 Individualized Risk Factor Analysis</h4>', unsafe_allow_html=True)
        st.caption("Which factors contributed most to this specific prediction?")
        
        coeffs = model.coef_[0]
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_final.columns
        
        # Map raw names to readable names
        readable_map = {
            'log_MLR': 'MLR (Inflammation)',
            'log_CRP': 'CRP (Inflammation)',
            'log_triglycerides': 'Triglycerides (Lipids)',
            'log_NLR': 'NLR (Inflammation)',
            'IJVC': 'Hx of IJV Cannulation',
            'sex': 'Sex',
            'log_MLR*log_CRP': 'Interaction: MLR x CRP',
            'log_MLR*log_triglycerides': 'Interaction: MLR x TG',
            'log_MLR*log_NLR': 'Interaction: MLR x NLR',
            # Add other potential interactions if they exist, or generic fallback
        }
        
        contributions = []
        for name, coef, val in zip(feature_names, coeffs, X_scaled[0]):
            readable_name = readable_map.get(name, name)
            # Calculate contribution (coef * value)
            contrib_val = coef * val
            contributions.append({'Risk Factor': readable_name, 'Impact': contrib_val})
            
        df_contrib = pd.DataFrame(contributions).sort_values(by='Impact', ascending=False, key=abs)
        
        # Color coding for chart
        df_contrib['Type'] = df_contrib['Impact'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
        
        # Altair Bar Chart
        c = alt.Chart(df_contrib.head(6)).mark_bar().encode(
            x=alt.X('Impact', title='Contribution to Risk Score'),
            y=alt.Y('Risk Factor', sort='-x', title=None),
            color=alt.Color('Type', scale=alt.Scale(domain=['Increases Risk', 'Decreases Risk'], range=['#e74c3c', '#2ecc71'])),
            tooltip=['Risk Factor', 'Impact', 'Type']
        ).properties(height=300)
        
        st.altair_chart(c, use_container_width=True)
        
        st.info("💡 **Note**: Positive bars (Red) indicate factors that are pushing the risk **higher** for this patient. Negative bars (Green) are protective factors.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug Info:", X_final)

else:
    # Placeholder when no prediction made
    st.info("👈 Please enter patient data in the sidebar and click 'Predict Risk' to start assessment.")
    
    # Add some visual fluff
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        st.markdown("""
        ### Key Features
        *   **Extreme Minimalism**: Uses only 6 variables.
        *   **High Utility**: Validated by NRI & DCA.
        *   **Inflammation-Centric**: Incorporates MLR & NLR.
        """)
    with col_demo2:
        st.markdown("""
        ### Target Population
        *   Maintenance Hemodialysis Patients
        *   Autogenous AVF
        *   Dialysis Vintage ≥ 3 months
        """)

# ---------------------------------------------------------
# 5. Variable Definitions
# ---------------------------------------------------------
st.divider()
with st.expander("ℹ️ Variable Definitions & Clinical Explanations (变量说明)", expanded=False):
    st.markdown("""
    | Variable (变量名) | Full Name (全称) | Description (Description / 说明) |
    | :--- | :--- | :--- |
    | **IJVC** | **Ipsilateral Internal Jugular Vein Cannulation** | **English**: History of central venous catheterization on the same side as the AVF. A key risk factor for central venous stenosis.<br>**Chinese**: **动静脉内瘘同侧颈内静脉置管史**。指患者在建立内瘘的一侧，既往是否进行过颈内静脉置管（透析导管）。这是导致中心静脉狭窄的重要危险因素。 |
    | **MLR** | **Monocyte-to-Lymphocyte Ratio** | **English**: Ratio of monocyte count to lymphocyte count. Reflects the balance between innate and adaptive immunity; high values are associated with vascular inflammation.<br>**Chinese**: **单核细胞与淋巴细胞比值**。反映全身炎症反应与免疫状态的平衡，高值通常与血管内膜增生和炎症相关。 |
    | **NLR** | **Neutrophil-to-Lymphocyte Ratio** | **English**: Ratio of neutrophil count to lymphocyte count. A classic marker of systemic inflammation and stress response.<br>**Chinese**: **中性粒细胞与淋巴细胞比值**。经典的系统性炎症指标，反映机体炎症及应激状态。 |
    | **CRP** | **C-Reactive Protein** | **English**: An acute-phase protein synthesized by the liver, serving as a sensitive marker of systemic inflammation.<br>**Chinese**: **C-反应蛋白**。肝脏合成的急性时相反应蛋白，反映体内系统性炎症水平的敏感指标。 |
    | **Triglycerides** | **Triglycerides** | **English**: A type of lipid in the blood. Abnormal lipid metabolism may impair vascular endothelial function.<br>**Chinese**: **甘油三酯**。血液中的一种脂质。脂代谢异常可能损害血管内皮功能并促进动脉粥样硬化。 |
    | **Sex** | **Gender** | **English**: Biological sex. Differences in vessel diameter and hormonal profiles may influence AVF patency.<br>**Chinese**: **性别**。解剖结构（如血管直径）和激素水平的差异可能影响内瘘通畅率及成熟结局。 |
    """)

# ---------------------------------------------------------
# 6. Footer / Disclaimer
# ---------------------------------------------------------
st.divider()
st.markdown("""
<small>
**Disclaimer:** This tool is for research and educational purposes only. 
It is based on the "Extreme Minimalist Model" developed in a single-center retrospective study.
Results should not replace professional clinical judgment.
</small>
""", unsafe_allow_html=True)
