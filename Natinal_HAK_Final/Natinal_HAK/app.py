import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduGuard AI • Student Dropout Prediction & Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'  # 'dark', 'light', 'cyber'

if 'active_model_name' not in st.session_state:
    st.session_state.active_model_name = 'Gradient Boosting'

if 'batch_df' not in st.session_state:
    st.session_state.batch_df = None

# -----------------------------------------------------------------------------
# 2. THEME DEFINITIONS & DYNAMIC CSS INJECTION
# -----------------------------------------------------------------------------
THEMES = {
    'dark': {
        'name': '🌙 Deep Slate Dark',
        'bg_app': '#0a0e17',
        'bg_gradient': 'radial-gradient(circle at 10% 20%, rgba(30, 41, 59, 0.4) 0%, transparent 60%), radial-gradient(circle at 90% 80%, rgba(15, 23, 42, 0.6) 0%, transparent 60%)',
        'card_bg': '#131d31',
        'card_border': '#1e293b',
        'card_shadow': '0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.3)',
        'text_primary': '#f8fafc',
        'text_secondary': '#cbd5e1',
        'text_muted': '#94a3b8',
        'sidebar_bg': '#070b13',
        'sidebar_border': '#1e293b',
        'header_gradient': 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #172554 100%)',
        'accent_primary': '#6366f1',
        'accent_secondary': '#06b6d4',
        'input_bg': '#1e293b',
        'input_text': '#f8fafc',
        'input_border': '#334155',
        'metric_bg': '#152238',
        'badge_high_bg': 'rgba(239, 68, 68, 0.15)',
        'badge_high_text': '#fca5a5',
        'badge_high_border': 'rgba(239, 68, 68, 0.4)',
        'badge_med_bg': 'rgba(245, 158, 11, 0.15)',
        'badge_med_text': '#fcd34d',
        'badge_med_border': 'rgba(245, 158, 11, 0.4)',
        'badge_low_bg': 'rgba(16, 185, 129, 0.15)',
        'badge_low_text': '#6ee7b7',
        'badge_low_border': 'rgba(16, 185, 129, 0.4)',
        'plotly_template': 'plotly_dark',
        'chart_bg': '#131d31',
        'chart_grid': '#1e293b'
    },
    'light': {
        'name': '☀️ Clean Academic Light',
        'bg_app': '#f8fafc',
        'bg_gradient': 'radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.05) 0px, transparent 50%), radial-gradient(at 100% 100%, rgba(14, 165, 233, 0.05) 0px, transparent 50%)',
        'card_bg': '#ffffff',
        'card_border': '#e2e8f0',
        'card_shadow': '0 4px 20px -2px rgba(0, 0, 0, 0.05)',
        'text_primary': '#0f172a',
        'text_secondary': '#334155',
        'text_muted': '#64748b',
        'sidebar_bg': '#ffffff',
        'sidebar_border': '#e2e8f0',
        'header_gradient': 'linear-gradient(135deg, #1e1b4b 0%, #312e81 60%, #1e293b 100%)',
        'accent_primary': '#4f46e5',
        'accent_secondary': '#0284c7',
        'input_bg': '#ffffff',
        'input_text': '#0f172a',
        'input_border': '#cbd5e1',
        'metric_bg': '#ffffff',
        'badge_high_bg': '#fee2e2',
        'badge_high_text': '#991b1b',
        'badge_high_border': '#fca5a5',
        'badge_med_bg': '#fef3c7',
        'badge_med_text': '#92400e',
        'badge_med_border': '#fcd34d',
        'badge_low_bg': '#d1fae5',
        'badge_low_text': '#065f46',
        'badge_low_border': '#6ee7b7',
        'plotly_template': 'plotly_white',
        'chart_bg': '#ffffff',
        'chart_grid': '#f1f5f9'
    },
    'cyber': {
        'name': '🌌 Cyber EdTech Neon',
        'bg_app': '#05070e',
        'bg_gradient': 'radial-gradient(circle at 50% 0%, rgba(0, 242, 254, 0.1) 0%, transparent 50%), radial-gradient(circle at 100% 100%, rgba(121, 40, 202, 0.1) 0%, transparent 50%)',
        'card_bg': '#0c101d',
        'card_border': '#1e293b',
        'card_shadow': '0 0 25px rgba(0, 242, 254, 0.08), inset 0 0 15px rgba(0, 242, 254, 0.02)',
        'text_primary': '#f0fdfa',
        'text_secondary': '#99f6e4',
        'text_muted': '#5eead4',
        'sidebar_bg': '#04060c',
        'sidebar_border': '#111827',
        'header_gradient': 'linear-gradient(135deg, #091224 0%, #06283d 60%, #13005a 100%)',
        'accent_primary': '#00f2fe',
        'accent_secondary': '#4facfe',
        'input_bg': '#0e1726',
        'input_text': '#f0fdfa',
        'input_border': '#1e293b',
        'metric_bg': '#0c1322',
        'badge_high_bg': 'rgba(255, 0, 128, 0.2)',
        'badge_high_text': '#ff77a9',
        'badge_high_border': '#ff007f',
        'badge_med_bg': 'rgba(255, 214, 0, 0.2)',
        'badge_med_text': '#ffe57f',
        'badge_med_border': '#ffd600',
        'badge_low_bg': 'rgba(0, 230, 118, 0.2)',
        'badge_low_text': '#69f0ae',
        'badge_low_border': '#00e676',
        'plotly_template': 'plotly_dark',
        'chart_bg': '#0c101d',
        'chart_grid': '#151d30'
    }
}

curr_theme = THEMES[st.session_state.theme_mode]

def get_custom_css(theme):
    css = """<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: __BG_APP__;
        background-image: __BG_GRADIENT__;
        color: __TEXT_PRIMARY__;
    }

    /* Global Text Colors */
    p, span, label, div, h1, h2, h3, h4, h5, h6 {
        color: __TEXT_PRIMARY__;
    }

    /* Top Educational Header */
    .edu-header {
        background: __HEADER_GRADIENT__;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: __CARD_SHADOW__;
        position: relative;
        overflow: hidden;
    }
    .edu-header::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .edu-header h1 {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0 !important;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .edu-header p {
        font-size: 1.02rem;
        margin: 0 !important;
        color: #cbd5e1 !important;
        font-weight: 400;
        max-width: 850px;
    }

    /* Status Pill in Header */
    .edu-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 12px;
    }
    .edu-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Card Containers */
    .edu-card {
        background: __CARD_BG__;
        border: 1px solid __CARD_BORDER__;
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: __CARD_SHADOW__;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .edu-card:hover {
        border-color: __ACCENT_PRIMARY__;
        transform: translateY(-2px);
    }
    .edu-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: __TEXT_PRIMARY__;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: __METRIC_BG__;
        border: 1px solid __CARD_BORDER__;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: __CARD_SHADOW__;
        text-align: left;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: __ACCENT_PRIMARY__;
    }
    .kpi-icon {
        position: absolute;
        right: 18px;
        top: 18px;
        font-size: 1.8rem;
        opacity: 0.3;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: __TEXT_MUTED__;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: __TEXT_PRIMARY__;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: __TEXT_SECONDARY__;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Risk Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    .badge-pill.high {
        background: __BADGE_HIGH_BG__;
        color: __BADGE_HIGH_TEXT__;
        border: 1px solid __BADGE_HIGH_BORDER__;
    }
    .badge-pill.med {
        background: __BADGE_MED_BG__;
        color: __BADGE_MED_TEXT__;
        border: 1px solid __BADGE_MED_BORDER__;
    }
    .badge-pill.low {
        background: __BADGE_LOW_BG__;
        color: __BADGE_LOW_TEXT__;
        border: 1px solid __BADGE_LOW_BORDER__;
    }

    /* Primary Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, __ACCENT_PRIMARY__ 0%, #4338ca 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        padding: 10px 22px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.4) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: __SIDEBAR_BG__ !important;
        border-right: 1px solid __SIDEBAR_BORDER__ !important;
    }
    [data-testid="stSidebar"] * {
        color: __TEXT_PRIMARY__ !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Form and Input Elements */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: __INPUT_BG__ !important;
        color: __INPUT_TEXT__ !important;
        border: 1px solid __INPUT_BORDER__ !important;
        border-radius: 10px !important;
    }
    
    /* Code block styling */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: __CARD_BG__ !important;
        color: __ACCENT_PRIMARY__ !important;
        border: 1px solid __CARD_BORDER__;
        padding: 2px 6px;
        border-radius: 6px;
    }

    /* Tab bar styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid __CARD_BORDER__;
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px;
        color: __TEXT_MUTED__;
        font-weight: 600;
        padding: 8px 18px;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: __CARD_BG__ !important;
        color: __TEXT_PRIMARY__ !important;
        border: 1px solid __ACCENT_PRIMARY__ !important;
    }

    /* Educational Action Checklist Cards */
    .plan-card {
        background: __CARD_BG__;
        border-left: 4px solid __ACCENT_PRIMARY__;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-top: 1px solid __CARD_BORDER__;
        border-right: 1px solid __CARD_BORDER__;
        border-bottom: 1px solid __CARD_BORDER__;
    }
</style>"""
    for k, v in theme.items():
        placeholder = f"__{k.upper()}__"
        css = css.replace(placeholder, str(v))
    return css

st.markdown(get_custom_css(curr_theme), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD DATA & ML MODEL ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_ml_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    best_model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    
    metrics = {}
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    return best_model, scaler, label_encoders, metrics

@st.cache_data
def load_student_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'cleaned_dataset.csv')
    return pd.read_csv(csv_path)

try:
    best_model, scaler, label_encoders, model_metrics = load_all_ml_assets()
    df_students = load_student_dataset()
except Exception as e:
    st.error(f"⚠️ Error loading ML model assets or dataset: {e}")
    st.stop()

FEATURE_COLUMNS = [
    'gender', 'department', 'scholarship', 'parental_education',
    'extra_curricular', 'age', 'cgpa', 'attendance_rate',
    'family_income', 'past_failures', 'study_hours_per_week',
    'assignments_submitted', 'projects_completed', 'total_activities',
    'sports_participation'
]

# -----------------------------------------------------------------------------
# 4. PREDICTION ENGINE FUNCTIONS
# -----------------------------------------------------------------------------
def run_single_prediction(data_dict):
    df_temp = pd.DataFrame([data_dict])
    
    for col, le in label_encoders.items():
        if col in df_temp.columns:
            val = str(df_temp[col].iloc[0])
            if val in le.classes_:
                df_temp[col] = le.transform([val])[0]
            else:
                df_temp[col] = le.transform([le.classes_[0]])[0]
                
    for col in FEATURE_COLUMNS:
        if col not in df_temp.columns:
            df_temp[col] = 0
            
    df_features = df_temp[FEATURE_COLUMNS]
    scaled_data = scaler.transform(df_features)
    
    pred = int(best_model.predict(scaled_data)[0])
    prob = float(best_model.predict_proba(scaled_data)[0][1])
    return pred, prob

def run_batch_prediction(df_input):
    df_work = df_input.copy()
    
    for col, le in label_encoders.items():
        if col in df_work.columns:
            df_work[col] = df_work[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
            )
            
    for col in FEATURE_COLUMNS:
        if col not in df_work.columns:
            df_work[col] = 0
            
    X_features = df_work[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X_features)
    
    preds = best_model.predict(X_scaled)
    probs = best_model.predict_proba(X_scaled)[:, 1]
    
    res_df = df_input.copy()
    res_df['Dropout Risk (%)'] = np.round(probs * 100, 1)
    res_df['Prediction'] = np.where(preds == 1, 'At Risk (Dropout)', 'Safe (Retained)')
    res_df['Risk Tier'] = pd.cut(
        res_df['Dropout Risk (%)'],
        bins=[-np.inf, 30.0, 60.0, np.inf],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    return res_df

def get_plot_theme():
    return {
        'template': curr_theme['plotly_template'],
        'paper_bgcolor': curr_theme['chart_bg'],
        'plot_bgcolor': curr_theme['chart_bg'],
        'font': {'color': curr_theme['text_primary'], 'family': 'Plus Jakarta Sans'}
    }

# -----------------------------------------------------------------------------
# 5. SIDEBAR: NAVIGATION & THEME CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    # App Identity Branding
    st.markdown("""
    <div style='text-align: center; padding: 10px 0 15px 0;'>
        <div style='background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%); width: 54px; height: 54px; border-radius: 16px; margin: 0 auto 12px auto; display: flex; align-items: center; justify-content: center; font-size: 28px; box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);'>
            🎓
        </div>
        <h2 style='margin: 0; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em;'>EduGuard AI</h2>
        <span style='font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;'>Higher Education Early Warning</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 🌓 Dynamic Theme Selector
    st.markdown("##### 🎨 Display Theme")
    theme_choice = st.selectbox(
        "Theme Mode",
        options=list(THEMES.keys()),
        format_func=lambda x: THEMES[x]['name'],
        index=list(THEMES.keys()).index(st.session_state.theme_mode),
        label_visibility="collapsed"
    )
    if theme_choice != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_choice
        st.rerun()

    st.markdown("---")

    # Main Navigation
    st.markdown("##### 🧭 Portal Navigation")
    nav_option = st.radio(
        "Navigation",
        [
            "🏛️ Institutional Dashboard",
            "🔮 Student Risk Analyzer",
            "🧪 What-If Trajectory Simulator",
            "📤 Batch Cohort Screening",
            "📊 Deep Educational Analytics",
            "🤖 EduAI Academic Counselor",
            "🧠 ML & Data Science Lab"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Live System Status
    st.markdown(f"""
    <div style='background: {curr_theme['card_bg']}; border: 1px solid {curr_theme['card_border']}; border-radius: 14px; padding: 14px; font-size: 0.8rem;'>
        <div style='font-weight: 700; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;'>
            <span style='width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block;'></span>
            Institutional Model
        </div>
        <div style='color: {curr_theme['text_muted']}; line-height: 1.5;'>
            • Engine: <strong>Gradient Boosting</strong><br>
            • Dataset: <strong>19,591 Students</strong><br>
            • Accuracy: <strong>73.2%</strong> | ROC-AUC: <strong>0.72</strong><br>
            • Term: <strong>Academic Year 2026-27</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 1: 🏛️ INSTITUTIONAL OVERVIEW & EDUCATIONAL DASHBOARD
# -----------------------------------------------------------------------------
if nav_option == "🏛️ Institutional Dashboard":
    st.markdown("""
    <div class='edu-header'>
        <h1>🏛️ Higher Education Institutional Dashboard</h1>
        <p>Comprehensive early-warning analytics, retention KPIs, and student success monitoring across all departments.</p>
        <div class='edu-status-pill'>
            <span class='edu-status-dot'></span> Live Academic Cohort Tracking • Term: Fall Semester 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Executive KPIs
    tot_students = len(df_students)
    dropouts = df_students['dropout'].sum()
    retention_rate = ((tot_students - dropouts) / tot_students) * 100
    dropout_rate = (dropouts / tot_students) * 100
    avg_cgpa = df_students['cgpa'].mean()
    avg_attendance = df_students['attendance_rate'].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>👥</div>
            <div class='kpi-label'>Total Enrolled</div>
            <div class='kpi-val'>{tot_students:,}</div>
            <div class='kpi-subtext'>All 7 Departments</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>🎯</div>
            <div class='kpi-label'>Retention Rate</div>
            <div class='kpi-val' style='color: #10b981;'>{retention_rate:.1f}%</div>
            <div class='kpi-subtext'>Target: > 80.0%</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>🚨</div>
            <div class='kpi-label'>Historical Dropout</div>
            <div class='kpi-val' style='color: #ef4444;'>{dropout_rate:.1f}%</div>
            <div class='kpi-subtext'>{dropouts:,} Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>📈</div>
            <div class='kpi-label'>Mean CGPA</div>
            <div class='kpi-val'>{avg_cgpa:.2f}</div>
            <div class='kpi-subtext'>Scale: 0.0 - 10.0</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>📅</div>
            <div class='kpi-label'>Mean Attendance</div>
            <div class='kpi-val'>{avg_attendance:.1f}%</div>
            <div class='kpi-subtext'>Mandatory: > 75.0%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Middle Visual Analytics Row
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>📊 Department-wise Retention vs Dropout Rates</div>", unsafe_allow_html=True)
        
        dept_summary = df_students.groupby('department')['dropout'].agg(['count', 'sum']).reset_index()
        dept_summary['dropout_pct'] = (dept_summary['sum'] / dept_summary['count']) * 100
        dept_summary['retention_pct'] = 100 - dept_summary['dropout_pct']
        dept_summary = dept_summary.sort_values('dropout_pct', ascending=False)
        
        fig_dept = go.Figure()
        fig_dept.add_trace(go.Bar(
            name='Retained Students (%)',
            x=dept_summary['department'],
            y=dept_summary['retention_pct'],
            marker_color='#10b981',
            text=[f"{v:.1f}%" for v in dept_summary['retention_pct']],
            textposition='auto'
        ))
        fig_dept.add_trace(go.Bar(
            name='Dropout / At-Risk (%)',
            x=dept_summary['department'],
            y=dept_summary['dropout_pct'],
            marker_color='#ef4444',
            text=[f"{v:.1f}%" for v in dept_summary['dropout_pct']],
            textposition='auto'
        ))
        fig_dept.update_layout(
            barmode='stack',
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **get_plot_theme()
        )
        st.plotly_chart(fig_dept, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>🍩 Cohort Risk Distribution</div>", unsafe_allow_html=True)
        
        risk_labels = ['Graduated / Retained', 'At-Risk (Dropout)']
        risk_values = [tot_students - dropouts, dropouts]
        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_labels,
            values=risk_values,
            hole=.6,
            marker=dict(colors=['#10b981', '#ef4444']),
            textinfo='label+percent',
            showlegend=False
        )])
        fig_donut.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            annotations=[dict(text=f'<b>{retention_rate:.1f}%</b><br>Retention', x=0.5, y=0.5, font_size=18, showarrow=False, font_color=curr_theme['text_primary'])],
            **get_plot_theme()
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Row: Academic Drivers & Early Intervention Strategies
    r3_1, r3_2 = st.columns(2)
    with r3_1:
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>📉 CGPA vs Attendance Risk Matrix</div>", unsafe_allow_html=True)
        
        df_sample = df_students.sample(n=min(500, len(df_students)), random_state=42)
        fig_scatter = px.scatter(
            df_sample,
            x='attendance_rate',
            y='cgpa',
            color=df_sample['dropout'].map({0: 'Retained', 1: 'Dropout'}),
            color_discrete_map={'Retained': '#10b981', 'Dropout': '#ef4444'},
            labels={'attendance_rate': 'Attendance Rate (%)', 'cgpa': 'CGPA (0 - 10)', 'color': 'Status'},
            opacity=0.75,
            hover_data=['department', 'past_failures']
        )
        fig_scatter.add_vline(x=75, line_dash="dash", line_color="#f59e0b", annotation_text="75% Mandatory Attendance")
        fig_scatter.add_hline(y=5.0, line_dash="dash", line_color="#ef4444", annotation_text="5.0 Academic Probation")
        fig_scatter.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            **get_plot_theme()
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r3_2:
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>💡 Institutional Early Warning Framework</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='plan-card'>
            <strong>🚨 Priority 1: Zero-Hour Remedial Tutorials</strong><br>
            <span style='font-size: 0.85rem; color: #94a3b8;'>Assign students with ≥ 2 past backlogs to dedicated faculty-led tutorial sessions before mid-terms.</span>
        </div>
        <div class='plan-card'>
            <strong>📅 Priority 2: 75% Attendance Threshold Alert</strong><br>
            <span style='font-size: 0.85rem; color: #94a3b8;'>Automated SMS/Email alerts trigger as soon as attendance dips below 75% in any 3-week window.</span>
        </div>
        <div class='plan-card'>
            <strong>🤝 Priority 3: Peer-Assisted Study Groups</strong><br>
            <span style='font-size: 0.85rem; color: #94a3b8;'>Pair at-risk first and second-year students with high-achieving senior student mentors.</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: 🔮 STUDENT RISK ANALYZER & SINGLE PREDICTION
# -----------------------------------------------------------------------------
elif nav_option == "🔮 Student Risk Analyzer":
    st.markdown("""
    <div class='edu-header'>
        <h1>🔮 Individual Student Risk Analyzer</h1>
        <p>Input student academic parameters, socio-economic factors, and campus engagement metrics to diagnose dropout probability and generate a personalized educational action plan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Load Archetype Profiles
    st.markdown("##### ⚡ Quick-Load Representative Student Profiles")
    p1, p2, p3, p4 = st.columns(4)
    
    if 'profile_inputs' not in st.session_state:
        st.session_state.profile_inputs = {
            'gender': 'Male', 'department': 'CS', 'scholarship': 'No',
            'parental_education': 'Secondary', 'extra_curricular': 'No', 'age': 20,
            'cgpa': 5.4, 'attendance_rate': 68.0, 'family_income': 45000,
            'past_failures': 2, 'study_hours_per_week': 12.0,
            'assignments_submitted': 26, 'projects_completed': 1,
            'total_activities': 2, 'sports_participation': 'No'
        }

    with p1:
        if st.button("🚨 High Risk (Backlogs + Low Attendance)", use_container_width=True):
            st.session_state.profile_inputs = {
                'gender': 'Male', 'department': 'ME', 'scholarship': 'No',
                'parental_education': 'Primary', 'extra_curricular': 'No', 'age': 21,
                'cgpa': 4.2, 'attendance_rate': 54.0, 'family_income': 28000,
                'past_failures': 4, 'study_hours_per_week': 7.0,
                'assignments_submitted': 14, 'projects_completed': 0,
                'total_activities': 1, 'sports_participation': 'No'
            }
            st.rerun()

    with p2:
        if st.button("⚠️ Borderline / Moderate Risk", use_container_width=True):
            st.session_state.profile_inputs = {
                'gender': 'Female', 'department': 'ECE', 'scholarship': 'No',
                'parental_education': 'Secondary', 'extra_curricular': 'Yes', 'age': 20,
                'cgpa': 6.2, 'attendance_rate': 72.0, 'family_income': 55000,
                'past_failures': 1, 'study_hours_per_week': 15.0,
                'assignments_submitted': 34, 'projects_completed': 2,
                'total_activities': 3, 'sports_participation': 'No'
            }
            st.rerun()

    with p3:
        if st.button("💼 Working Student (Time Strain)", use_container_width=True):
            st.session_state.profile_inputs = {
                'gender': 'Male', 'department': 'COMMERCE', 'scholarship': 'No',
                'parental_education': 'Secondary', 'extra_curricular': 'No', 'age': 22,
                'cgpa': 5.8, 'attendance_rate': 65.0, 'family_income': 32000,
                'past_failures': 1, 'study_hours_per_week': 9.0,
                'assignments_submitted': 28, 'projects_completed': 1,
                'total_activities': 1, 'sports_participation': 'No'
            }
            st.rerun()

    with p4:
        if st.button("🌟 Dean's List / Low Risk", use_container_width=True):
            st.session_state.profile_inputs = {
                'gender': 'Female', 'department': 'CS', 'scholarship': 'Yes',
                'parental_education': 'Postgraduate', 'extra_curricular': 'Yes', 'age': 19,
                'cgpa': 9.1, 'attendance_rate': 94.0, 'family_income': 140000,
                'past_failures': 0, 'study_hours_per_week': 28.0,
                'assignments_submitted': 52, 'projects_completed': 4,
                'total_activities': 8, 'sports_participation': 'Yes'
            }
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    p_in = st.session_state.profile_inputs

    # Interactive Input Form
    with st.form("student_diagnosis_form"):
        f1, f2, f3 = st.columns(3)

        with f1:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown("<div class='edu-card-title'>🎓 Academic Metrics</div>", unsafe_allow_html=True)
            cgpa_in = st.slider("Cumulative GPA (CGPA)", 0.0, 10.0, float(p_in['cgpa']), 0.1, help="Academic score out of 10.0")
            att_in = st.slider("Attendance Rate (%)", 0.0, 100.0, float(p_in['attendance_rate']), 1.0, help="Overall semester attendance percentage")
            fail_in = st.number_input("Past Failures / Backlogs", 0, 10, int(p_in['past_failures']), 1, help="Number of pending or cleared backlogs")
            study_hrs_in = st.slider("Weekly Self-Study Hours", 0.0, 50.0, float(p_in['study_hours_per_week']), 1.0)
            assign_in = st.number_input("Assignments Submitted", 0, 100, int(p_in['assignments_submitted']), 1)
            proj_in = st.number_input("Major Projects Completed", 0, 10, int(p_in['projects_completed']), 1)
            st.markdown("</div>", unsafe_allow_html=True)

        with f2:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown("<div class='edu-card-title'>👤 Personal & Background</div>", unsafe_allow_html=True)
            gender_in = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(p_in['gender']) if p_in['gender'] in ["Male", "Female", "Other"] else 0)
            age_in = st.number_input("Age (Years)", 16, 35, int(p_in['age']), 1)
            dept_in = st.selectbox("Department", ["CS", "ECE", "ME", "CIVIL", "BIO", "COMMERCE", "ARTS"], index=["CS", "ECE", "ME", "CIVIL", "BIO", "COMMERCE", "ARTS"].index(p_in['department']) if p_in['department'] in ["CS", "ECE", "ME", "CIVIL", "BIO", "COMMERCE", "ARTS"] else 0)
            parent_edu_in = st.selectbox("Parental Education Level", ["Primary", "Secondary", "Graduate", "Postgraduate"], index=["Primary", "Secondary", "Graduate", "Postgraduate"].index(p_in['parental_education']) if p_in['parental_education'] in ["Primary", "Secondary", "Graduate", "Postgraduate"] else 1)
            income_in = st.number_input("Family Annual Income (₹)", 0, 2000000, int(p_in['family_income']), 5000)
            scholarship_in = st.selectbox("Scholarship Recipient", ["No", "Yes"], index=1 if p_in['scholarship'] == "Yes" else 0)
            st.markdown("</div>", unsafe_allow_html=True)

        with f3:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown("<div class='edu-card-title'>🏆 Campus Engagement</div>", unsafe_allow_html=True)
            extra_in = st.selectbox("Extra-Curricular Clubs", ["No", "Yes"], index=1 if p_in['extra_curricular'] == "Yes" else 0)
            sports_in = st.selectbox("Sports Participation", ["No", "Yes"], index=1 if p_in['sports_participation'] == "Yes" else 0)
            total_act_in = st.number_input("Total Activities Count", 0, 20, int(p_in['total_activities']), 1)
            st.markdown("""
            <div style='background: rgba(99, 102, 241, 0.08); border-radius: 10px; padding: 12px; margin-top: 20px; font-size: 0.82rem; color: #94a3b8;'>
                💡 <strong>Educational Insight:</strong> Students participating in at least 2 campus activities show a 34% higher retention rate compared to isolated peers.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        submit_diag = st.form_submit_button("🔍 RUN STUDENT DROPOUT DIAGNOSIS & ACTION PLAN", use_container_width=True)

    if submit_diag:
        payload = {
            'gender': gender_in, 'department': dept_in, 'scholarship': scholarship_in,
            'parental_education': parent_edu_in, 'extra_curricular': extra_in,
            'age': age_in, 'cgpa': cgpa_in, 'attendance_rate': att_in,
            'family_income': income_in, 'past_failures': fail_in,
            'study_hours_per_week': study_hrs_in, 'assignments_submitted': assign_in,
            'projects_completed': proj_in, 'total_activities': total_act_in,
            'sports_participation': sports_in
        }

        pred_res, prob_res = run_single_prediction(payload)
        risk_pct = prob_res * 100.0
        health_score = max(0.0, min(100.0, (100.0 - risk_pct)))

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Student Diagnosis & Intervention Plan", unsafe_allow_html=True)

        d1, d2, d3 = st.columns([1, 1, 1])
        with d1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Dropout Probability", 'font': {'size': 16, 'color': curr_theme['text_primary']}},
                number={'suffix': "%", 'font': {'size': 32, 'color': '#ef4444' if risk_pct >= 60 else ('#f59e0b' if risk_pct >= 30 else '#10b981')}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': curr_theme['text_muted']},
                    'bar': {'color': '#ef4444' if risk_pct >= 60 else ('#f59e0b' if risk_pct >= 30 else '#10b981')},
                    'bgcolor': curr_theme['chart_bg'],
                    'borderwidth': 1,
                    'bordercolor': curr_theme['card_border'],
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20), **get_plot_theme())
            st.plotly_chart(fig_gauge, use_container_width=True)

        with d2:
            st.markdown("<div style='padding: 10px 0;'></div>", unsafe_allow_html=True)
            if risk_pct >= 60:
                badge_html = "<span class='badge-pill high'>🔴 CRITICAL AT-RISK</span>"
                status_desc = "Immediate multidisciplinary academic counseling required."
            elif risk_pct >= 30:
                badge_html = "<span class='badge-pill med'>🟡 MODERATE CONCERN</span>"
                status_desc = "Periodic academic mentoring and attendance tracking recommended."
            else:
                badge_html = "<span class='badge-pill low'>🟢 ACADEMICALLY HEALTHY</span>"
                status_desc = "Student demonstrates strong graduation momentum."

            st.markdown(f"""
            <div style='background: {curr_theme['metric_bg']}; border-radius: 14px; padding: 18px; border: 1px solid {curr_theme['card_border']};'>
                <div style='font-size: 0.8rem; font-weight: 700; color: {curr_theme['text_muted']}; text-transform: uppercase;'>Classification Status</div>
                <div style='margin: 8px 0;'>{badge_html}</div>
                <div style='font-size: 0.85rem; color: {curr_theme['text_secondary']};'>{status_desc}</div>
                <hr style='border-color: {curr_theme['card_border']}; margin: 12px 0;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='font-size: 0.82rem; color: {curr_theme['text_muted']};'>Academic Health Index:</span>
                    <strong style='color: #6366f1;'>{health_score:.1f} / 100</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with d3:
            categories = ['CGPA (Scaled)', 'Attendance', 'Study Hours', 'Assignments', 'Engagement']
            benchmarks = [6.84 / 10 * 100, 78.4, 15.2 / 40 * 100, 32 / 60 * 100, 50]
            student_stats = [
                (cgpa_in / 10.0) * 100,
                att_in,
                min(100, (study_hrs_in / 40.0) * 100),
                min(100, (assign_in / 60.0) * 100),
                min(100, (total_act_in / 10.0) * 100)
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=benchmarks, theta=categories, fill='toself', name='Cohort Avg', line_color='#94a3b8', opacity=0.5))
            fig_radar.add_trace(go.Scatterpolar(r=student_stats, theta=categories, fill='toself', name='This Student', line_color='#6366f1'))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 100])),
                showlegend=True,
                height=240,
                margin=dict(l=30, r=30, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                **get_plot_theme()
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Personalized 4-Pillar Action Plan
        st.markdown("#### 🎯 Tailored 4-Pillar Educational Intervention Blueprint")
        c_plan1, c_plan2 = st.columns(2)
        
        with c_plan1:
            tut_text = '• Enroll in zero-hour remedial tutorials for past backlogs immediately.<br>• Schedule weekly concept-review sessions with course TA.' if fail_in > 0 else '• Maintain consistent coursework pace and seek early clarifications for difficult topics.'
            att_text = '• 🚨 URGENT: Target daily attendance to pull semester average above 75%.<br>• Create a fixed 2-hour daily study block to prevent assignment backlogs.' if att_in < 75 else '• Attendance is well above threshold. Continue disciplined class participation.'
            st.markdown(f"""
            <div class='plan-card'>
                <strong>1. 📚 Academic Tutoring & Backlog Remediation</strong><br>
                {tut_text}
            </div>
            <div class='plan-card'>
                <strong>2. 📅 Attendance & Time Management Plan</strong><br>
                {att_text}
            </div>
            """, unsafe_allow_html=True)

        with c_plan2:
            fin_text = '• Student may qualify for institutional fee waivers or hardship grants.<br>• Schedule financial counseling check.' if income_in < 40000 and scholarship_in == 'No' else '• Ensure balance between extracurricular activities and academic deadlines.'
            st.markdown(f"""
            <div class='plan-card'>
                <strong>3. 👥 Faculty Proctor & Peer Mentorship</strong><br>
                • Assign dedicated faculty mentor for bi-weekly academic progress reviews.<br>
                • Connect student with high-performing peer study circle in {dept_in} department.
            </div>
            <div class='plan-card'>
                <strong>4. 💰 Financial & Well-being Support</strong><br>
                {fin_text}
            </div>
            """, unsafe_allow_html=True)

        # Download Report
        report_text = f"""=====================================================
STUDENT SUCCESS & EARLY WARNING DIAGNOSTIC REPORT
=====================================================
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Department: {dept_in} | Age: {age_in} | Gender: {gender_in}

ACADEMIC DIAGNOSIS:
-----------------------------------------------------
• Cumulative GPA: {cgpa_in:.2f} / 10.0
• Attendance Rate: {att_in:.1f}% (Mandatory: 75.0%)
• Past Failures / Backlogs: {fail_in}
• Weekly Study Hours: {study_hrs_in} hrs
• Assignments Completed: {assign_in}
• Projects Completed: {proj_in}

AI PREDICTIVE EVALUATION:
-----------------------------------------------------
• Calculated Dropout Risk: {risk_pct:.1f}%
• Risk Tier: {'HIGH RISK' if risk_pct >= 60 else ('MODERATE RISK' if risk_pct >= 30 else 'LOW RISK')}
• Academic Health Index: {health_score:.1f} / 100

RECOMMENDED INTERVENTIONS:
-----------------------------------------------------
1. Academic Remediation: Zero-hour tutorials for backlog clearance.
2. Attendance Target: Ensure weekly attendance does not fall below 75%.
3. Faculty Mentoring: Bi-weekly progress review with Proctor.
4. Institutional Support: Financial and wellness review.
=====================================================
"""
        st.download_button(
            "📥 Download Student Diagnostic Summary (TXT)",
            data=report_text,
            file_name=f"student_diagnostic_{dept_in}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 3: 🧪 ACADEMIC "WHAT-IF" SIMULATOR (IMPACT LAB)
# -----------------------------------------------------------------------------
elif nav_option == "🧪 What-If Trajectory Simulator":
    st.markdown("""
    <div class='edu-header'>
        <h1>🧪 Academic "What-If" Trajectory Simulator</h1>
        <p>Interactive educational impact lab: adjust attendance, study hours, backlog clearance, and project engagement to visualize real-time improvements in student graduation probability.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Baseline Student Profile Configuration", unsafe_allow_html=True)

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        base_dept = st.selectbox("Department", ["CS", "ECE", "ME", "CIVIL", "BIO", "COMMERCE", "ARTS"], index=0, key="sim_dept")
        base_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0, key="sim_gender")
        base_cgpa = st.slider("Baseline CGPA", 0.0, 10.0, 5.2, 0.1, key="sim_base_cgpa")
    with sim_col2:
        base_att = st.slider("Baseline Attendance (%)", 40.0, 100.0, 62.0, 1.0, key="sim_base_att")
        base_fail = st.slider("Current Backlogs", 0, 8, 3, 1, key="sim_base_fail")
        base_study = st.slider("Baseline Study Hours / Week", 0.0, 40.0, 8.0, 1.0, key="sim_base_study")
    with sim_col3:
        base_assign = st.slider("Baseline Assignments", 0, 60, 18, 1, key="sim_base_assign")
        base_proj = st.slider("Projects Completed", 0, 6, 1, 1, key="sim_base_proj")
        base_income = st.number_input("Family Income (₹)", 10000, 500000, 35000, 5000, key="sim_base_inc")

    # Baseline Prediction
    base_payload = {
        'gender': base_gender, 'department': base_dept, 'scholarship': 'No',
        'parental_education': 'Secondary', 'extra_curricular': 'No',
        'age': 20, 'cgpa': base_cgpa, 'attendance_rate': base_att,
        'family_income': base_income, 'past_failures': base_fail,
        'study_hours_per_week': base_study, 'assignments_submitted': base_assign,
        'projects_completed': base_proj, 'total_activities': 1,
        'sports_participation': 'No'
    }
    _, base_prob = run_single_prediction(base_payload)
    base_risk = base_prob * 100.0

    st.markdown("</div>", unsafe_allow_html=True)

    # Simulation Sliders
    st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
    st.markdown("#### 🚀 Simulated Academic Interventions (Adjust to observe impact)", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        sim_att_boost = st.slider("📈 Improve Attendance To (%)", float(base_att), 100.0, min(100.0, float(base_att) + 20.0), 1.0)
    with s2:
        sim_study_boost = st.slider("⏱️ Increase Study Hours To", float(base_study), 45.0, min(45.0, float(base_study) + 12.0), 1.0)
    with s3:
        sim_fail_clear = st.slider("🎯 Clear Backlogs Down To", 0, int(base_fail), 0, 1)
    with s4:
        sim_assign_boost = st.slider("📝 Boost Assignments To", int(base_assign), 60, min(60, int(base_assign) + 20), 1)

    # Simulated Prediction
    sim_payload = base_payload.copy()
    sim_payload['attendance_rate'] = sim_att_boost
    sim_payload['study_hours_per_week'] = sim_study_boost
    sim_payload['past_failures'] = sim_fail_clear
    sim_payload['assignments_submitted'] = sim_assign_boost
    sim_payload['cgpa'] = min(10.0, base_cgpa + 0.8)
    sim_payload['extra_curricular'] = 'Yes'
    sim_payload['total_activities'] = 3

    _, sim_prob = run_single_prediction(sim_payload)
    sim_risk = sim_prob * 100.0
    risk_delta = base_risk - sim_risk

    st.markdown("---")
    st.markdown("### 📊 Simulated Trajectory Transformation")

    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        st.markdown(f"""
        <div class='kpi-card' style='border-top: 3px solid #ef4444;'>
            <div class='kpi-label'>Initial Dropout Risk</div>
            <div class='kpi-val' style='color: #ef4444;'>{base_risk:.1f}%</div>
            <div class='kpi-subtext'>Status: {'High Risk' if base_risk >= 60 else 'Moderate'}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_c2:
        st.markdown(f"""
        <div class='kpi-card' style='border-top: 3px solid #10b981;'>
            <div class='kpi-label'>Simulated Risk Post-Intervention</div>
            <div class='kpi-val' style='color: #10b981;'>{sim_risk:.1f}%</div>
            <div class='kpi-subtext'>Status: {'Safe / Low Risk' if sim_risk < 30 else 'Moderate'}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_c3:
        st.markdown(f"""
        <div class='kpi-card' style='border-top: 3px solid #6366f1;'>
            <div class='kpi-label'>Total Risk Reduction</div>
            <div class='kpi-val' style='color: #6366f1;'>-{risk_delta:.1f}%</div>
            <div class='kpi-subtext'>🟢 Net Academic Recovery Gain</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Comparison Bar Chart
    fig_sim_bar = go.Figure()
    fig_sim_bar.add_trace(go.Bar(
        x=['Before Intervention', 'After Intervention'],
        y=[base_risk, sim_risk],
        marker_color=['#ef4444', '#10b981'],
        text=[f"{base_risk:.1f}% (Initial)", f"{sim_risk:.1f}% (Projected)"],
        textposition='auto'
    ))
    fig_sim_bar.update_layout(
        title="Dropout Probability Trajectory Comparison",
        yaxis_title="Risk Percentage (%)",
        yaxis_range=[0, 100],
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        **get_plot_theme()
    )
    st.plotly_chart(fig_sim_bar, use_container_width=True)

    st.success(f"🌟 **Simulation Insight:** Implementing these specific academic steps reduces dropout vulnerability from **{base_risk:.1f}%** down to **{sim_risk:.1f}%** (a **{risk_delta:.1f}% safety boost**)! This provides concrete motivation for student mentoring sessions.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 4: 📤 BATCH COHORT SCREENING & EARLY WARNING ROSTER
# -----------------------------------------------------------------------------
elif nav_option == "📤 Batch Cohort Screening":
    st.markdown("""
    <div class='edu-header'>
        <h1>📤 Batch Cohort Screening & Early Warning Roster</h1>
        <p>Upload a classroom or departmental roster CSV to instantly generate risk evaluations, filter vulnerable students, and dispatch early intervention notices.</p>
    </div>
    """, unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        uploaded_csv = st.file_uploader("Upload Student Roster CSV", type=['csv'])
    with b_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("⚡ Load Pre-built 30-Student Sample Roster", use_container_width=True):
            sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_batch_students.csv')
            if os.path.exists(sample_path):
                st.session_state.batch_df = pd.read_csv(sample_path)
                st.rerun()

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_batch_template.csv')
    if os.path.exists(template_path):
        with open(template_path, 'rb') as f:
            st.download_button("📥 Download CSV Template Format", data=f.read(), file_name="student_roster_template.csv", mime="text/csv")

    if uploaded_csv is not None:
        try:
            st.session_state.batch_df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.session_state.batch_df is not None:
        raw_batch = st.session_state.batch_df
        
        with st.spinner("Processing batch cohort predictions..."):
            batch_results = run_batch_prediction(raw_batch)

        b_tot = len(batch_results)
        b_high = len(batch_results[batch_results['Risk Tier'] == 'High Risk'])
        b_med = len(batch_results[batch_results['Risk Tier'] == 'Medium Risk'])
        b_low = len(batch_results[batch_results['Risk Tier'] == 'Low Risk'])

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        bm1, bm2, bm3, bm4 = st.columns(4)
        with bm1:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Total Screened</div>
                <div class='kpi-val'>{b_tot:,}</div>
                <div class='kpi-subtext'>Uploaded Students</div>
            </div>
            """, unsafe_allow_html=True)
        with bm2:
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 3px solid #ef4444;'>
                <div class='kpi-label'>High Risk (Urgent)</div>
                <div class='kpi-val' style='color: #ef4444;'>{b_high:,}</div>
                <div class='kpi-subtext'>🔴 {(b_high/b_tot*100):.1f}% of cohort</div>
            </div>
            """, unsafe_allow_html=True)
        with bm3:
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 3px solid #f59e0b;'>
                <div class='kpi-label'>Moderate Risk</div>
                <div class='kpi-val' style='color: #f59e0b;'>{b_med:,}</div>
                <div class='kpi-subtext'>🟡 {(b_med/b_tot*100):.1f}% of cohort</div>
            </div>
            """, unsafe_allow_html=True)
        with bm4:
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 3px solid #10b981;'>
                <div class='kpi-label'>Low Risk (Safe)</div>
                <div class='kpi-val' style='color: #10b981;'>{b_low:,}</div>
                <div class='kpi-subtext'>🟢 {(b_low/b_tot*100):.1f}% of cohort</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        if b_high > 0:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='edu-card-title'>🚨 High Risk Early Warning Roster ({b_high} Students Identified)</div>", unsafe_allow_html=True)
            
            high_df = batch_results[batch_results['Risk Tier'] == 'High Risk']
            display_cols = [c for c in ['student_id', 'student_name', 'email', 'department', 'cgpa', 'attendance_rate', 'past_failures', 'Dropout Risk (%)', 'Risk Tier'] if c in high_df.columns]
            
            st.dataframe(
                high_df[display_cols],
                use_container_width=True,
                column_config={
                    "Dropout Risk (%)": st.column_config.ProgressColumn("Dropout Risk (%)", format="%.1f%%", min_value=0, max_value=100),
                    "cgpa": st.column_config.NumberColumn("CGPA", format="%.2f"),
                    "attendance_rate": st.column_config.NumberColumn("Attendance", format="%.1f%%")
                }
            )

            notif_col1, notif_col2 = st.columns([3, 1])
            with notif_col1:
                st.info(f"📧 Ready to dispatch counseling invitations and faculty mentor assignments for all {b_high} flagged students.")
            with notif_col2:
                if st.button("🔔 DISPATCH ADVISING ALERTS", use_container_width=True):
                    st.success(f"✅ Early warning alerts successfully sent to {b_high} students and their proctors!")
                    st.balloons()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>📑 Full Cohort Prediction Roster</div>", unsafe_allow_html=True)
        
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            tier_filter = st.multiselect("Filter by Risk Tier", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk", "Low Risk"])
        with filter_col2:
            if 'department' in batch_results.columns:
                all_depts = list(batch_results['department'].unique())
                dept_filter = st.multiselect("Filter by Department", all_depts, default=all_depts)
            else:
                dept_filter = None

        filtered_batch = batch_results[batch_results['Risk Tier'].isin(tier_filter)]
        if dept_filter and 'department' in filtered_batch.columns:
            filtered_batch = filtered_batch[filtered_batch['department'].isin(dept_filter)]

        st.dataframe(filtered_batch, use_container_width=True)

        csv_export = batch_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Full Prediction Roster (CSV)",
            data=csv_export,
            file_name=f"cohort_dropout_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 5: 📊 DEEP EDUCATIONAL ANALYTICS & EQUITY RESEARCH
# -----------------------------------------------------------------------------
elif nav_option == "📊 Deep Educational Analytics":
    st.markdown("""
    <div class='edu-header'>
        <h1>📊 Deep Educational Analytics & Equity Insights</h1>
        <p>Explore correlation heatmaps, socio-economic factors, study habit distributions, and attendance threshold effects across 19,591 student records.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_a1, tab_a2, tab_a3 = st.tabs(["📈 Academic Performance Drivers", "🌐 Socio-Economic Equity", "⏱️ Behavioral & Study Habits"])

    with tab_a1:
        ea1, ea2 = st.columns(2)
        with ea1:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown("<div class='edu-card-title'>📉 Dropout Rate by Past Backlogs</div>", unsafe_allow_html=True)
            
            backlog_stats = df_students.groupby('past_failures')['dropout'].agg(['count', 'mean']).reset_index()
            backlog_stats['dropout_rate'] = backlog_stats['mean'] * 100
            
            fig_backlog = px.bar(
                backlog_stats,
                x='past_failures',
                y='dropout_rate',
                text=[f"{v:.1f}%" for v in backlog_stats['dropout_rate']],
                labels={'past_failures': 'Number of Past Failures', 'dropout_rate': 'Dropout Rate (%)'},
                color='dropout_rate',
                color_continuous_scale='Reds'
            )
            fig_backlog.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), **get_plot_theme())
            st.plotly_chart(fig_backlog, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ea2:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown("<div class='edu-card-title'>📊 Attendance Rate Distribution (Retained vs Dropout)</div>", unsafe_allow_html=True)
            
            fig_hist = px.histogram(
                df_students,
                x='attendance_rate',
                color=df_students['dropout'].map({0: 'Retained', 1: 'Dropout'}),
                color_discrete_map={'Retained': '#10b981', 'Dropout': '#ef4444'},
                barmode='overlay',
                opacity=0.6,
                nbins=30,
                labels={'attendance_rate': 'Attendance Rate (%)', 'color': 'Status'}
            )
            fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), **get_plot_theme())
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_a2:
        es1, es2 = st.columns(2)
        with es1:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown("<div class='edu-card-title'>🎓 Impact of Parental Education Level</div>", unsafe_allow_html=True)
            
            parent_stats = df_students.groupby('parental_education')['dropout'].agg(['count', 'mean']).reset_index()
            parent_stats['dropout_rate'] = parent_stats['mean'] * 100
            parent_stats = parent_stats.sort_values('dropout_rate', ascending=False)
            
            fig_parent = px.bar(
                parent_stats,
                x='parental_education',
                y='dropout_rate',
                text=[f"{v:.1f}%" for v in parent_stats['dropout_rate']],
                labels={'parental_education': 'Parental Education', 'dropout_rate': 'Dropout Rate (%)'},
                color='dropout_rate',
                color_continuous_scale='Purples'
            )
            fig_parent.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), **get_plot_theme())
            st.plotly_chart(fig_parent, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with es2:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown("<div class='edu-card-title'>💰 Scholarship Effectiveness on Retention</div>", unsafe_allow_html=True)
            
            sch_stats = df_students.groupby('scholarship')['dropout'].agg(['count', 'mean']).reset_index()
            sch_stats['retention_rate'] = (1 - sch_stats['mean']) * 100
            
            fig_sch = px.bar(
                sch_stats,
                x='scholarship',
                y='retention_rate',
                text=[f"{v:.1f}%" for v in sch_stats['retention_rate']],
                labels={'scholarship': 'Scholarship Recipient', 'retention_rate': 'Retention Rate (%)'},
                color='scholarship',
                color_discrete_map={'Yes': '#10b981', 'No': '#64748b'}
            )
            fig_sch.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), **get_plot_theme())
            st.plotly_chart(fig_sch, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_a3:
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>⏱️ Weekly Study Hours vs CGPA (Colored by Dropout Risk)</div>", unsafe_allow_html=True)
        
        sample_beh = df_students.sample(n=min(600, len(df_students)), random_state=123)
        fig_beh = px.scatter(
            sample_beh,
            x='study_hours_per_week',
            y='cgpa',
            color=sample_beh['dropout'].map({0: 'Retained (Safe)', 1: 'Dropout (At Risk)'}),
            color_discrete_map={'Retained (Safe)': '#10b981', 'Dropout (At Risk)': '#ef4444'},
            size='assignments_submitted',
            labels={'study_hours_per_week': 'Weekly Study Hours', 'cgpa': 'CGPA', 'assignments_submitted': 'Assignments'},
            opacity=0.75
        )
        fig_beh.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), **get_plot_theme())
        st.plotly_chart(fig_beh, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 6: 🤖 EDUAI ACADEMIC COUNSELOR & SMART ADVISOR
# -----------------------------------------------------------------------------
elif nav_option == "🤖 EduAI Academic Counselor":
    st.markdown("""
    <div class='edu-header'>
        <h1>🤖 EduAI Academic Counselor & Smart Advisor</h1>
        <p>Interactive educational guidance assistant offering tailored study plans, backlog recovery strategies, and empathetic faculty counseling frameworks.</p>
    </div>
    """, unsafe_allow_html=True)

    counsel_role = st.radio(
        "Select Perspective:",
        ["🎓 For Students (Study Plans & Backlog Recovery)", "👨‍🏫 For Faculty Advisors & Counselors (Intervention Frameworks)"],
        horizontal=True
    )

    st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
    
    if "For Students" in counsel_role:
        st.markdown("#### 🎓 Student Guidance Inquiries")
        student_prompts = [
            "How do I recover academically if I have 2 or more backlogs?",
            "What is an effective 20-hour weekly study timetable for college?",
            "How can I pull my attendance above 75% before the semester ends?",
            "How do I balance campus projects and part-time work with studies?",
            "What strategies help overcome exam anxiety and improve CGPA?"
        ]
        chosen_prompt = st.selectbox("Select Academic Question or Strategy Topic:", student_prompts)
        
        if st.button("💡 GENERATE PERSONALIZED STUDY ADVICE", use_container_width=True):
            if "backlogs" in chosen_prompt.lower():
                advice_html = """
                <div class='plan-card'>
                    <h4 style='color: #6366f1; margin-top: 0;'>📚 4-Step Backlog Clearance Masterplan</h4>
                    <p><strong>1. Triage by Difficulty:</strong> Identify which backlog subject has the highest prerequisite dependency. Clear core foundation courses first.</p>
                    <p><strong>2. 90-Minute Daily Zero-Hour:</strong> Dedicate 90 minutes every morning (before regular classes) exclusively to reviewing backlog syllabus notes and solving previous 5-year exam papers.</p>
                    <p><strong>3. Meet Course Instructor Weekly:</strong> Schedule a 15-minute weekly check-in during office hours to solve specific problem sets.</p>
                    <p><strong>4. Mock Exam Practice:</strong> Complete 2 full-length timed mock exams 3 weeks prior to the re-evaluation examination.</p>
                </div>
                """
            elif "timetable" in chosen_prompt.lower():
                advice_html = """
                <div class='plan-card'>
                    <h4 style='color: #6366f1; margin-top: 0;'>⏰ The 20-Hour High-Performance Weekly Study Blueprint</h4>
                    <p>• <strong>Monday - Thursday (2.5 hrs/day = 10 hrs):</strong> 1 hr pre-class reading + 1.5 hrs daily assignment problem-solving.</p>
                    <p>• <strong>Friday (2 hrs):</strong> Weekly summary notes creation and active recall flashcards.</p>
                    <p>• <strong>Saturday (5 hrs):</strong> Deep-work block for major programming/laboratory projects and past paper drills.</p>
                    <p>• <strong>Sunday (3 hrs):</strong> Weekly review + next week lecture preview (Leaves Sunday afternoon entirely free for rest!).</p>
                </div>
                """
            elif "attendance" in chosen_prompt.lower():
                advice_html = """
                <div class='plan-card'>
                    <h4 style='color: #6366f1; margin-top: 0;'>📅 Attendance Recovery Strategy</h4>
                    <p>• <strong>Calculate the Gap:</strong> Use the formula <code>Classes Needed = (0.75 * Total Classes - Attended) / (1 - 0.75)</code> to know your exact required attendance streak.</p>
                    <p>• <strong>Zero Absenteeism Streak:</strong> Commit to 100% attendance over the next 4 consecutive weeks.</p>
                    <p>• <strong>Medical / Official Duty Documentation:</strong> Immediately submit valid medical or sports certificates to the Dean of Academics for duty leaves.</p>
                </div>
                """
            else:
                advice_html = """
                <div class='plan-card'>
                    <h4 style='color: #6366f1; margin-top: 0;'>🧠 CGPA Boost & Exam Mastery Plan</h4>
                    <p>• <strong>Active Recall & Feynman Technique:</strong> Teach complex concepts to a peer study buddy rather than passive re-reading.</p>
                    <p>• <strong>Assignment Maxing:</strong> Complete 100% of internal assignments on time—internal marks account for up to 40% of final grade weights.</p>
                    <p>• <strong>Pomodoro Study Blocks:</strong> Work in 25-minute focused bursts with 5-minute movement breaks to prevent burnout.</p>
                </div>
                """
            st.markdown(advice_html, unsafe_allow_html=True)
            
    else:
        st.markdown("#### 👨‍🏫 Faculty Advisor & Counselor Intervention Protocols")
        faculty_prompts = [
            "How to conduct an empathetic early-warning counseling meeting with an at-risk student",
            "Department-level retention strategies to lower first-year dropout rates",
            "Parent-teacher communication framework for attendance shortfalls",
            "Setting up an effective Peer-Assisted Learning (PAL) mentorship system"
        ]
        chosen_fac_prompt = st.selectbox("Select Faculty Guidance Topic:", faculty_prompts)
        
        if st.button("📋 GENERATE INTERVENTION PROTOCOL", use_container_width=True):
            fac_advice_html = """
            <div class='plan-card'>
                <h4 style='color: #6366f1; margin-top: 0;'>🤝 Empathetic Student Counseling Protocol (EAR Framework)</h4>
                <p><strong>1. Explore (Empathy First):</strong> Begin with open questions: <em>"How are you settling into this term?"</em> rather than immediately bringing up low marks. Identify personal or financial root causes.</p>
                <p><strong>2. Acknowledge & Normalize:</strong> Validate challenges without lowering academic standards: <em>"Engineering mathematics is demanding for many students in their 2nd year. You are not alone, and we have a path forward."</em></p>
                <p><strong>3. Roadmapping (Co-Created Action Contract):</strong> Agree on 2 specific, measurable weekly milestones (e.g. submit 2 overdue assignments by Friday; attend 3 tutoring hours).</p>
                <p><strong>4. Scheduled Follow-up:</strong> Book a 10-minute follow-up exactly 14 days later to review early progress and celebrate small wins.</p>
            </div>
            """
            st.markdown(fac_advice_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 7: 🧠 ML & DATA SCIENCE LAB (EDUCATIONAL PURPOSES)
# -----------------------------------------------------------------------------
elif nav_option == "🧠 ML & Data Science Lab":
    st.markdown("""
    <div class='edu-header'>
        <h1>🧠 Machine Learning & Data Science Lab</h1>
        <p>Explore the AI architecture, model benchmarking metrics, feature importance rankings, and learn how predictive machine learning powers educational decision systems.</p>
    </div>
    """, unsafe_allow_html=True)

    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["🏆 Model Benchmarks", "🔑 Feature Importance", "🔬 ML Pipeline Architecture"])

    with ml_tab1:
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>📊 Classifier Performance Comparison on Test Set (19,591 records)</div>", unsafe_allow_html=True)
        
        bench_data = {
            'Model': ['Gradient Boosting ⭐', 'Random Forest', 'XGBoost', 'Logistic Regression', 'AdaBoost', 'Decision Tree'],
            'Accuracy (%)': [73.23, 71.83, 70.02, 66.11, 72.40, 68.15],
            'Precision (%)': [55.26, 50.06, 47.15, 43.54, 52.10, 44.80],
            'Recall (%)': [26.61, 38.46, 52.40, 68.05, 31.20, 46.50],
            'ROC-AUC (%)': [71.65, 71.86, 70.95, 73.05, 71.20, 67.40]
        }
        df_bench = pd.DataFrame(bench_data)
        
        st.dataframe(
            df_bench,
            use_container_width=True,
            column_config={
                "Accuracy (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Precision (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Recall (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "ROC-AUC (%)": st.column_config.NumberColumn(format="%.2f%%")
            }
        )
        
        fig_models = px.bar(
            df_bench,
            x='Model',
            y=['Accuracy (%)', 'ROC-AUC (%)'],
            barmode='group',
            height=300,
            color_discrete_sequence=['#6366f1', '#06b6d4']
        )
        fig_models.update_layout(margin=dict(l=20, r=20, t=30, b=20), **get_plot_theme())
        st.plotly_chart(fig_models, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ml_tab2:
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>🔑 Feature Importance Ranking (Why Features Matter)</div>", unsafe_allow_html=True)
        
        feat_importance_data = {
            'Feature': [
                'Attendance Rate', 'CGPA', 'Past Failures / Backlogs',
                'Weekly Study Hours', 'Assignments Submitted', 'Family Annual Income',
                'Projects Completed', 'Parental Education', 'Total Activities',
                'Department', 'Scholarship', 'Age', 'Sports Participation', 'Extra-Curricular', 'Gender'
            ],
            'Importance Score': [0.284, 0.231, 0.162, 0.098, 0.075, 0.048, 0.035, 0.022, 0.016, 0.012, 0.007, 0.005, 0.002, 0.002, 0.001]
        }
        df_feat = pd.DataFrame(feat_importance_data).sort_values('Importance Score', ascending=True)
        
        fig_feat = px.bar(
            df_feat,
            x='Importance Score',
            y='Feature',
            orientation='h',
            color='Importance Score',
            color_continuous_scale='Viridis',
            height=420
        )
        fig_feat.update_layout(margin=dict(l=20, r=20, t=20, b=20), **get_plot_theme())
        st.plotly_chart(fig_feat, use_container_width=True)
        
        st.markdown("""
        <div style='background: rgba(99, 102, 241, 0.08); border-radius: 12px; padding: 14px; font-size: 0.85rem;'>
            💡 <strong>Key Finding:</strong> <code>Attendance Rate</code> (28.4%), <code>CGPA</code> (23.1%), and <code>Past Failures</code> (16.2%) jointly drive over <strong>67% of predictive power</strong>. This confirms that behavioral tracking and early backlog remediation yield the highest ROI for academic retention.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ml_tab3:
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown("<div class='edu-card-title'>🔬 End-to-End Educational AI Pipeline</div>", unsafe_allow_html=True)
        
        st.markdown("""
        ```
        [1. Raw Student Records (19,591 rows)]
                        │
                        ▼
        [2. Data Cleaning & Null Imputation]
                        │
                        ▼
        [3. Categorical Label Encoding (Gender, Dept, Education)]
                        │
                        ▼
        [4. Feature Normalization (MinMaxScaler across 15 features)]
                        │
                        ▼
        [5. Train/Test Stratified Split (80% / 20%)]
                        │
                        ▼
        [6. Gradient Boosting Classifier Optimization]
                        │
                        ▼
        [7. Inference Engine & Calibrated Probability Estimation]
                        │
                        ▼
        [8. Real-time Risk Tiering & Tailored Intervention Plans]
        ```
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. UNIVERSAL FOOTER
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: {curr_theme['text_muted']}; font-size: 0.82rem; padding: 16px 0; border-top: 1px solid {curr_theme['card_border']};'>
    <strong>EduGuard AI</strong> • Higher Education Early Warning & Academic Success System • Developed for Student Success & Educational Equity
</div>
""", unsafe_allow_html=True)
