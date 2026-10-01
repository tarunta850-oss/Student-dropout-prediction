import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from openai import OpenAI

# Page configuration - FORCE SIDEBAR TO BE ALWAYS VISIBLE
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Neo-Brutalism CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap');

    * {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #FFF5E1 0%, #FFE4E1 100%);
    }

    /* Headers */
    h1, h2, h3 {
        color: #000 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        text-shadow: 4px 4px 0px #FF6B6B;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #000 !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 700 !important;
        text-transform: uppercase;
        color: #000 !important;
    }

    /* Buttons */
    .stButton>button {
        background: #FFE66D !important;
        color: #000 !important;
        border: 4px solid #000 !important;
        border-radius: 0px !important;
        padding: 15px 30px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        box-shadow: 6px 6px 0px #000 !important;
        transition: all 0.1s ease !important;
    }

    .stButton>button:hover {
        transform: translate(2px, 2px) !important;
        box-shadow: 4px 4px 0px #000 !important;
        background: #FFD93D !important;
    }

    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background: #FFF !important;
        border: 3px solid #000 !important;
        border-radius: 0px !important;
        color: #000 !important;
        font-weight: 600 !important;
        box-shadow: 4px 4px 0px #000 !important;
    }

    /* Sidebar - ALWAYS VISIBLE */
    [data-testid="stSidebar"] {
        background: #4ECDC4 !important;
        border-right: 5px solid #000 !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }

    /* Sidebar collapse button - make it visible */
    [data-testid="collapsedControl"] {
        background: #FFE66D !important;
        border: 3px solid #000 !important;
        box-shadow: 4px 4px 0px #000 !important;
        color: #000 !important;
    }

    /* Force sidebar to stay open */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 300px !important;
        transform: none !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 300px !important;
        transform: none !important;
        margin-left: 0 !important;
    }

    [data-testid="stSidebar"] * {
        color: #000 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] label {
        font-size: 1.1rem !important;
        color: #000 !important;
        font-weight: 700 !important;
    }

    /* Fix radio button text */
    [data-testid="stSidebar"] .row-widget {
        background: rgba(255,255,255,0.8) !important;
        padding: 10px !important;
        border: 3px solid #000 !important;
        margin: 5px 0 !important;
    }

    /* Radio buttons - ensure visibility */
    [data-testid="stSidebar"] .st-emotion-cache-1gulkj5 {
        background: white !important;
        color: #000 !important;
    }

    /* Radio button labels */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #000 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }

    /* Radio options - UNIFORM SIZE */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: white !important;
        color: #000 !important;
        font-weight: 700 !important;
        padding: 15px 20px !important;
        border: 3px solid #000 !important;
        margin: 8px 0 !important;
        display: block !important;
        box-shadow: 4px 4px 0px #000 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        min-height: 50px !important;
        text-align: left !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #FFE66D !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px #000 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] [data-checked="true"] {
        background: #A8E6CF !important;
        font-weight: 900 !important;
    }

    /* Ensure radio button container is full width */
    [data-testid="stSidebar"] [role="radiogroup"] {
        width: 100% !important;
    }

    /* Cards/Containers */
    .stContainer, div[data-testid="stExpander"] {
        background: #FFF !important;
        border: 4px solid #000 !important;
        border-radius: 0px !important;
        box-shadow: 8px 8px 0px #000 !important;
        padding: 20px !important;
        margin: 10px 0px !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 4px solid #000 !important;
        box-shadow: 6px 6px 0px #000 !important;
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: #FF6B6B !important;
        border: 2px solid #000 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #FFF !important;
        border: 3px solid #000 !important;
        border-radius: 0px !important;
        color: #000 !important;
        font-weight: 700 !important;
        box-shadow: 4px 4px 0px #000 !important;
    }

    .stTabs [aria-selected="true"] {
        background: #A8E6CF !important;
    }

    /* Risk badges */
    .risk-high {
        background: #FF6B6B;
        color: #000;
        padding: 8px 20px;
        border: 3px solid #000;
        box-shadow: 4px 4px 0px #000;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        margin: 5px;
    }

    .risk-medium {
        background: #FFD93D;
        color: #000;
        padding: 8px 20px;
        border: 3px solid #000;
        box-shadow: 4px 4px 0px #000;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        margin: 5px;
    }

    .risk-low {
        background: #A8E6CF;
        color: #000;
        padding: 8px 20px;
        border: 3px solid #000;
        box-shadow: 4px 4px 0px #000;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        margin: 5px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #FFF !important;
        border: 4px dashed #000 !important;
        border-radius: 0px !important;
        padding: 20px !important;
    }

    /* Better text visibility - FORCE BLACK TEXT EVERYWHERE */
    p, span, div, label, h1, h2, h3, h4, h5, h6, a, li, td, th {
        color: #000 !important;
        font-weight: 600 !important;
    }

    /* Markdown text */
    .stMarkdown, .stMarkdown *, [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {
        color: #000 !important;
        font-weight: 600 !important;
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        border: 3px solid #000 !important;
        border-radius: 0px !important;
        box-shadow: 5px 5px 0px #000 !important;
        background: white !important;
    }

    .stAlert * {
        color: #000 !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #FFF !important;
        border: 3px solid #000 !important;
        box-shadow: 5px 5px 0px #000 !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }

    [data-testid="stChatMessage"] * {
        color: #000 !important;
    }

    /* Chat input visibility - BLACK TEXT ON WHITE BACKGROUND */
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea,
    .stChatInput input,
    .stChatInput textarea,
    [data-testid="stChatInputTextArea"],
    [data-baseweb="textarea"] textarea {
        color: #000 !important;
        background: white !important;
        font-weight: 600 !important;
        border: 3px solid #000 !important;
        padding: 10px !important;
    }

    /* Chat input placeholder text */
    [data-testid="stChatInput"] input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder,
    .stChatInput input::placeholder,
    .stChatInput textarea::placeholder,
    [data-baseweb="textarea"] textarea::placeholder {
        color: #666 !important;
        opacity: 0.7 !important;
    }

    /* Ensure ALL text elements are black with white/light backgrounds */
    * {
        color: #000 !important;
    }

    /* Force white/light backgrounds for all content areas */
    .main .block-container {
        padding: 2rem !important;
        background: white !important;
    }

    /* All data display elements */
    [data-testid="stText"], [data-testid="stCaption"], .stText, .stCaption {
        color: #000 !important;
        background: white !important;
        padding: 5px !important;
    }

    /* Dataframe text */
    .dataframe, .dataframe * {
        color: #000 !important;
        background: white !important;
    }

    /* Select box and dropdown text */
    [data-baseweb="select"] *, [data-baseweb="popover"] * {
        color: #000 !important;
        background: white !important;
    }

    /* Floating Chatbot Button */
    .floating-chat-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        background: #FFE66D !important;
        border: 4px solid #000 !important;
        border-radius: 50% !important;
        box-shadow: 6px 6px 0px #000 !important;
        cursor: pointer;
        z-index: 9998;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        transition: all 0.3s ease;
    }

    .floating-chat-btn:hover {
        transform: scale(1.1);
        box-shadow: 8px 8px 0px #000 !important;
    }

    /* Floating Chatbot Container */
    .floating-chat-container {
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 400px;
        max-height: 600px;
        background: #FFF !important;
        border: 5px solid #000 !important;
        box-shadow: 8px 8px 0px #000 !important;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        font-family: 'Space Grotesk', sans-serif;
    }

    .floating-chat-header {
        background: #4ECDC4 !important;
        border-bottom: 4px solid #000 !important;
        padding: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .floating-chat-header h3 {
        margin: 0;
        color: #000 !important;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 18px;
    }

    .floating-chat-close {
        background: #FF6B6B;
        border: 3px solid #000;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #000;
    }

    .floating-chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 15px;
        background: #FFF;
        max-height: 400px;
    }

    .floating-chat-input-container {
        border-top: 4px solid #000;
        padding: 15px;
        background: #F5F5F5;
    }

    /* Hide default Streamlit chat styling within floating chatbot */
    .floating-chat-container .stChatMessage {
        border: 3px solid #000 !important;
        margin-bottom: 10px;
        padding: 10px;
        box-shadow: 3px 3px 0px #000;
    }

    .floating-chat-container [data-testid="stChatInput"] {
        border: 3px solid #000 !important;
        box-shadow: 3px 3px 0px #000 !important;
    }
</style>

<script>
    // Force sidebar to be visible on page load
    window.addEventListener('load', function() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.transform = 'none';
            sidebar.style.minWidth = '300px';
        }
    });

    // Monitor and keep sidebar open
    const observer = new MutationObserver(function(mutations) {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.transform = 'none';
        }
    });

    // Start observing
    setTimeout(() => {
        const targetNode = document.body;
        observer.observe(targetNode, { attributes: true, childList: true, subtree: true });
    }, 100);
</script>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    return model, scaler, label_encoders

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_dataset.csv')

# Load everything
try:
    model, scaler, label_encoders = load_models()
    default_df = load_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# Initialize session state for dynamic data
if 'current_data' not in st.session_state:
    st.session_state.current_data = default_df.copy()
    st.session_state.data_source = "default"
    st.session_state.predictions_computed = False
    st.session_state.analysis_results = {}

# Use current data from session state
df = st.session_state.current_data

# Initialize OpenAI client
try:
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        OPENAI_ENABLED = True
    else:
        OPENAI_ENABLED = False
except Exception as e:
    OPENAI_ENABLED = False
    st.warning("OpenAI integration unavailable. Chatbot will use basic responses.")

# System context for chatbot - comprehensive knowledge about the system
SYSTEM_CONTEXT = """You are an AI assistant for a Student Dropout Prediction System. You have complete knowledge of the system and should provide helpful, contextual responses.

## SYSTEM OVERVIEW:
- **Purpose**: Predict student dropout risk and provide intervention recommendations
- **Model**: Logistic Regression with 73.82% accuracy, ROC AUC 0.7305
- **Dataset**: 19,591 student records
- **Features**: 15 input variables (academic, behavioral, socio-economic)

## KEY STATISTICS:
- Total Students: 19,591
- Dropout Rate: 28.20%
- Average CGPA: 6.8
- Average Attendance: 78%
- Students At Risk: 5,526

## TOP RISK FACTORS (SHAP Analysis):
1. **Past Failures** (0.484) - Strongest predictor of dropout
2. **Attendance Rate** (0.445) - Critical indicator, <70% is high risk
3. **CGPA** (0.400) - Academic performance, <6.0 means 3x higher risk
4. **Sports Participation** (0.142) - Strong protective effect
5. **Department** (0.050) - Some programs have higher dropout rates

## 15 INPUT FEATURES:
1. gender (Male/Female)
2. age (16-30 years)
3. department (Engineering, Science, Arts, etc.)
4. cgpa (0.0-10.0)
5. attendance_rate (0-100%)
6. study_hours_per_week (0-60 hours)
7. assignments_submitted (0-100)
8. past_failures (0-10)
9. extra_curricular (Yes/No)
10. sports_participation (Yes/No)
11. projects_completed (0-10)
12. family_income (₹0-₹200,000)
13. parental_education (None/High School/Bachelor/Master/PhD)
14. scholarship (Yes/No)
15. total_activities (0-20)

## RISK LEVELS:
- **LOW (<30%)**: Student on track, continue monitoring
- **MEDIUM (30-60%)**: Needs attention, implement support
- **HIGH (>60%)**: Immediate intervention required

## PAGES IN THE SYSTEM:
1. **Dashboard**: Overview with KPIs, department statistics, key insights
2. **Predict Single**: Form-based prediction for individual students with recommendations
3. **Predict Batch**: CSV upload for bulk predictions (1000+ records/second)
4. **Students List**: Browse all 19,591 students with search and filters
5. **Analytics**: Data visualizations, SHAP plots, correlations, distributions
6. **Simulator**: What-if scenarios to test intervention impact
7. **Chatbot**: AI guidance assistant (you!)

## KEY INSIGHTS:
- CGPA < 6.0 → 3x higher dropout risk
- Attendance < 70% → Critical warning sign
- Scholarship holders → 30% lower dropout risk
- Extra-curricular participation → 40% risk reduction
- Sports participation → Strong protective effect
- Past failures → Strongest single predictor
- Family income and parental education → Moderate impact

## INTERVENTION STRATEGIES:

### High-Risk Students (>60%):
- Immediate counseling session
- Contact parents/guardians
- Personalized academic support plan
- Daily attendance monitoring
- Mentorship program enrollment
- Financial aid assistance if needed
- Peer support group assignment

### Medium-Risk Students (30-60%):
- Bi-weekly check-ins
- Study skills workshop
- Time management training
- Connect with academic advisors
- Encourage club participation

### Low-Risk Students (<30%):
- Monthly monitoring
- Maintain engagement
- Leadership opportunities
- Recognition programs

## ACADEMIC SUPPORT:
- **CGPA Improvement**: Tutoring, study groups, time management, test-taking strategies
- **Attendance**: Identify barriers, flexible schedules, peer attendance buddies
- **Study Hours**: Study skills workshops, productive study techniques
- **Assignments**: Planning assistance, deadline management

## BEHAVIORAL SUPPORT:
- **Extra-curriculars**: Encourage joining 2-3 activities for engagement
- **Sports**: Highly beneficial for retention and stress relief
- **Projects**: Technical societies, hackathons, collaborative work
- **Activities**: Cultural clubs, student government, volunteer work

## FINANCIAL SUPPORT:
- Merit-based scholarships
- Need-based grants
- Part-time campus jobs
- Emergency financial aid
- Payment plan options
- Book rental programs
- Transportation assistance
- Meal plan support

## MODEL DETAILS:
- **Algorithm**: Logistic Regression (best performance among tested models)
- **Training Data**: 15,672 students (80% split)
- **Test Data**: 3,919 students (20% split)
- **Accuracy**: 73.82%
- **Precision**: High for both classes
- **Recall**: Balanced detection
- **ROC AUC**: 0.7305
- **Cross-Validation**: 5-fold CV for robust evaluation

## SYSTEM CAPABILITIES:
- Real-time single predictions (<100ms)
- Batch processing (1000+ records/second)
- SHAP explainability for transparency
- Risk stratification (Low/Medium/High)
- Personalized recommendations
- What-if scenario testing
- Department-wise analytics
- Student record search and filtering
- CSV export capabilities

## USAGE NOTES:
- The model analyzes all 15 features together
- Predictions are probabilistic (0-100%)
- Higher probability = higher dropout risk
- Recommendations are tailored to risk level
- Early intervention is most effective
- Multi-factor approach works best

When answering questions:
1. Provide specific, actionable guidance
2. Reference relevant statistics and insights
3. Suggest appropriate pages/features for user's needs
4. Explain risk factors and interventions clearly
5. Be empathetic and solution-focused
6. Draw from the system knowledge above
7. Guide users to relevant pages when applicable"""

# Feature columns
FEATURE_COLUMNS = ['gender', 'department', 'scholarship', 'parental_education',
                   'extra_curricular', 'age', 'cgpa', 'attendance_rate',
                   'family_income', 'past_failures', 'study_hours_per_week',
                   'assignments_submitted', 'projects_completed', 'total_activities',
                   'sports_participation']

def preprocess_input(data):
    """Preprocess input data"""
    df_input = pd.DataFrame([data])

    for col in label_encoders.keys():
        if col in df_input.columns:
            le = label_encoders[col]
            df_input[col] = df_input[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df_input[col] = le.transform(df_input[col])

    for col in FEATURE_COLUMNS:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[FEATURE_COLUMNS]
    scaled_data = scaler.transform(df_input)

    return scaled_data

def predict_dropout(data):
    """Make prediction"""
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0]
    return prediction, probability

def get_predictions_for_dataframe(data_df, data_source_id):
    """
    Compute predictions for entire dataframe with caching based on data source
    Returns: DataFrame with predictions and probabilities
    """
    # Check if predictions are already cached for this data source
    cache_key = f'predictions_cache_{data_source_id}'
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Compute predictions
    X = data_df.drop(['student_id', 'dropout'], axis=1, errors='ignore')
    for col in label_encoders.keys():
        if col in X.columns:
            le = label_encoders[col]
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col])

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Create result dataframe
    result_df = data_df.copy()
    result_df['prediction'] = predictions
    result_df['dropout_probability'] = probabilities
    result_df['risk_level'] = result_df['dropout_probability'].apply(
        lambda x: 'High' if x >= 0.6 else ('Medium' if x >= 0.3 else 'Low')
    )

    # Cache the result
    st.session_state[cache_key] = result_df
    return result_df

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='color: #000 !important; text-shadow: 3px 3px 0px #FFE66D;'>🎓 NAVIGATION</h1>", unsafe_allow_html=True)
    page = st.radio("SELECT PAGE:",
                    ["Dashboard", "Predict Single", "Predict Batch", "Students List",
                     "Analytics", "Simulator"],
                    label_visibility="visible")

    st.markdown("---")
    st.markdown("<h3 style='color: #000 !important;'>📊 QUICK STATS</h3>", unsafe_allow_html=True)

    # Show data source indicator
    if st.session_state.data_source == "uploaded":
        st.markdown("**🔄 Using Uploaded Data**", unsafe_allow_html=True)
    else:
        st.markdown("**📁 Using Default Dataset**", unsafe_allow_html=True)

    total_students = len(df)
    dropout_count = df['dropout'].sum()
    dropout_rate = (dropout_count / total_students) * 100

    st.metric("Total Students", f"{total_students:,}")
    st.metric("Dropout Rate", f"{dropout_rate:.1f}%")
    st.metric("At Risk", f"{dropout_count:,}")

# Main content
if page == "Dashboard":
    st.markdown("# 🎓 STUDENT DROPOUT PREDICTOR")
    st.markdown("### AI-POWERED EARLY WARNING SYSTEM")

    # Data source indicator
    if st.session_state.data_source == "uploaded":
        st.info("📊 **Currently analyzing YOUR uploaded data!** All statistics below reflect the CSV you uploaded.")
    else:
        st.info("📁 **Currently showing default dataset.** Upload CSV in 'Predict Batch' to analyze your own data.")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("TOTAL STUDENTS", f"{len(df):,}",
                 help="Total number of students in the system")

    with col2:
        actual_dropouts = df['dropout'].sum()
        st.metric("ACTUAL DROPOUTS", f"{actual_dropouts:,}",
                 help="Number of students who dropped out")

    with col3:
        dropout_rate = (actual_dropouts / len(df)) * 100
        st.metric("DROPOUT RATE", f"{dropout_rate:.1f}%",
                 help="Percentage of students who dropped out")

    with col4:
        avg_cgpa = df['cgpa'].mean()
        st.metric("AVG CGPA", f"{avg_cgpa:.2f}",
                 help="Average CGPA of all students")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### DROPOUT BY DEPARTMENT")
        dept_dropout = df.groupby('department')['dropout'].agg(['sum', 'count'])
        dept_dropout['rate'] = (dept_dropout['sum'] / dept_dropout['count']) * 100

        fig = px.bar(dept_dropout.reset_index(), x='department', y='rate',
                    color='rate', color_continuous_scale='Reds',
                    labels={'rate': 'Dropout Rate (%)', 'department': 'Department'})
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Space Grotesk", color="black", weight=700)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### CGPA DISTRIBUTION")
        fig = px.histogram(df, x='cgpa', color='dropout',
                          labels={'cgpa': 'CGPA', 'dropout': 'Dropout Status'},
                          color_discrete_map={0: '#A8E6CF', 1: '#FF6B6B'})
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Space Grotesk", color="black", weight=700)
        )
        st.plotly_chart(fig, use_container_width=True)

    # More charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ATTENDANCE VS DROPOUT")
        fig = px.box(df, x='dropout', y='attendance_rate',
                    color='dropout',
                    labels={'attendance_rate': 'Attendance Rate (%)', 'dropout': 'Dropout Status'},
                    color_discrete_map={0: '#A8E6CF', 1: '#FF6B6B'})
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Space Grotesk", color="black", weight=700)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### SCHOLARSHIP VS DROPOUT")
        scholar_dropout = df.groupby('scholarship')['dropout'].mean() * 100
        fig = px.bar(scholar_dropout.reset_index(), x='scholarship', y='dropout',
                    color='dropout', color_continuous_scale='RdYlGn_r',
                    labels={'dropout': 'Dropout Rate (%)', 'scholarship': 'Scholarship Status'})
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Space Grotesk", color="black", weight=700)
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Predict Single":
    st.markdown("# 🔮 SINGLE STUDENT PREDICTION")
    st.markdown("### Enter student details to predict dropout risk")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", df['gender'].unique())
            department = st.selectbox("Department", df['department'].unique())
            scholarship = st.selectbox("Scholarship", df['scholarship'].unique())
            parental_education = st.selectbox("Parental Education", df['parental_education'].unique())
            extra_curricular = st.selectbox("Extra Curricular", df['extra_curricular'].unique())

        with col2:
            age = st.number_input("Age", min_value=16, max_value=30, value=20)
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
            family_income = st.number_input("Family Income", min_value=0, max_value=1000000, value=50000, step=1000)
            past_failures = st.number_input("Past Failures", min_value=0, max_value=10, value=0)

        with col3:
            study_hours_per_week = st.number_input("Study Hours/Week", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            assignments_submitted = st.number_input("Assignments Submitted", min_value=0, max_value=100, value=40)
            projects_completed = st.number_input("Projects Completed", min_value=0, max_value=10, value=3)
            total_activities = st.number_input("Total Activities", min_value=0, max_value=20, value=5)
            sports_participation = st.selectbox("Sports Participation", df['sports_participation'].unique())

        submit = st.form_submit_button("PREDICT DROPOUT RISK", use_container_width=True)

    if submit:
        input_data = {
            'gender': gender,
            'department': department,
            'scholarship': scholarship,
            'parental_education': parental_education,
            'extra_curricular': extra_curricular,
            'age': age,
            'cgpa': cgpa,
            'attendance_rate': attendance_rate,
            'family_income': family_income,
            'past_failures': past_failures,
            'study_hours_per_week': study_hours_per_week,
            'assignments_submitted': assignments_submitted,
            'projects_completed': projects_completed,
            'total_activities': total_activities,
            'sports_participation': sports_participation
        }

        prediction, probability = predict_dropout(input_data)
        dropout_prob = probability[1] * 100

        st.markdown("---")
        st.markdown("## 📊 PREDICTION RESULTS")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("DROPOUT PROBABILITY", f"{dropout_prob:.1f}%")

        with col2:
            risk_level = "HIGH" if dropout_prob >= 60 else ("MEDIUM" if dropout_prob >= 30 else "LOW")
            risk_class = f"risk-{risk_level.lower()}"
            st.markdown(f"**RISK LEVEL**")
            st.markdown(f'<div class="{risk_class}">{risk_level} RISK</div>', unsafe_allow_html=True)

        with col3:
            status = "AT RISK ⚠️" if prediction == 1 else "SAFE ✓"
            st.metric("STATUS", status)

        # Progress bar
        st.markdown("### RISK METER")
        st.progress(dropout_prob / 100)

        # Recommendations
        st.markdown("### 💡 RECOMMENDATIONS")
        if dropout_prob >= 60:
            st.error("""
            **IMMEDIATE ACTION REQUIRED:**
            - Schedule urgent counseling session
            - Contact parents/guardians
            - Develop personalized intervention plan
            - Monitor daily attendance
            - Provide academic support resources
            """)
        elif dropout_prob >= 30:
            st.warning("""
            **ATTENTION NEEDED:**
            - Regular check-ins with student
            - Monitor academic performance
            - Encourage participation in activities
            - Provide study resources
            """)
        else:
            st.success("""
            **STUDENT ON TRACK:**
            - Continue monitoring progress
            - Encourage continued engagement
            - Maintain support systems
            """)

elif page == "Predict Batch":
    st.markdown("# 📤 BATCH PREDICTION")
    st.markdown("### Upload CSV file for multiple predictions")

    st.markdown("""
    **Required CSV columns:**
    - gender, department, scholarship, parental_education, extra_curricular
    - age, cgpa, attendance_rate, family_income, past_failures
    - study_hours_per_week, assignments_submitted, projects_completed
    - total_activities, sports_participation
    - **email** (required for sending notifications to high-risk students)
    """)

    uploaded_file = st.file_uploader("UPLOAD CSV FILE", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} students")

            st.markdown("### PREVIEW DATA")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("RUN PREDICTIONS & USE AS PRIMARY DATA", use_container_width=True, key="run_predictions_btn"):
                with st.spinner("Processing predictions..."):
                    results = []
                    batch_df_with_predictions = batch_df.copy()
                    predictions_list = []
                    probabilities_list = []

                    for idx, row in batch_df.iterrows():
                        try:
                            row_dict = row.to_dict()
                            prediction, probability = predict_dropout(row_dict)
                            dropout_prob = probability[1]

                            risk_level = "High" if dropout_prob >= 0.6 else ("Medium" if dropout_prob >= 0.3 else "Low")

                            # Get email from row - handle both 'email' and missing email column
                            email_value = 'N/A'
                            if 'email' in row.index and pd.notna(row['email']):
                                email_value = row['email']

                            result_entry = {
                                'student_id': row.get('student_id', idx + 1),
                                'prediction': 'Dropout' if prediction == 1 else 'No Dropout',
                                'dropout_probability': dropout_prob,
                                'risk_level': risk_level,
                                'email': email_value  # Always include email column
                            }

                            results.append(result_entry)

                            predictions_list.append(prediction)
                            probabilities_list.append(dropout_prob)

                        except Exception as e:
                            # Get email from row - handle both 'email' and missing email column
                            email_value = 'N/A'
                            if 'email' in row.index and pd.notna(row['email']):
                                email_value = row['email']

                            result_entry = {
                                'student_id': row.get('student_id', idx + 1),
                                'error': str(e),
                                'email': email_value  # Always include email column
                            }
                            results.append(result_entry)
                            predictions_list.append(0)
                            probabilities_list.append(0.0)

                    results_df = pd.DataFrame(results)

                    # Add predictions to batch_df and update session state
                    if 'student_id' not in batch_df_with_predictions.columns:
                        batch_df_with_predictions['student_id'] = range(1, len(batch_df_with_predictions) + 1)

                    # Ensure dropout column exists
                    if 'dropout' not in batch_df_with_predictions.columns:
                        batch_df_with_predictions['dropout'] = predictions_list

                    # Store high-risk students BEFORE rerun
                    high_risk_students = results_df[results_df['risk_level'] == 'High'].copy()

                    # Update session state to make this the primary data source
                    st.session_state.current_data = batch_df_with_predictions
                    st.session_state.data_source = "uploaded"
                    st.session_state.predictions_computed = True
                    st.session_state.predictions_df = results_df
                    st.session_state.high_risk_students = high_risk_students  # Store in session state
                    st.session_state.batch_df_original = batch_df.copy()  # Store original uploaded data
                    st.session_state.analysis_results = {
                        'total': len(results_df),
                        'high_risk': len(results_df[results_df['risk_level'] == 'High']),
                        'medium_risk': len(results_df[results_df['risk_level'] == 'Medium']),
                        'low_risk': len(results_df[results_df['risk_level'] == 'Low']),
                        'dropout_rate': (sum(predictions_list) / len(predictions_list)) * 100 if len(predictions_list) > 0 else 0,
                        'avg_cgpa': batch_df_with_predictions['cgpa'].mean() if 'cgpa' in batch_df_with_predictions.columns else 0,
                        'avg_attendance': batch_df_with_predictions['attendance_rate'].mean() if 'attendance_rate' in batch_df_with_predictions.columns else 0
                    }
                    st.session_state.show_results = True  # Flag to show results
                    st.rerun()  # Force rerun to show results

            # DISPLAY RESULTS IF PREDICTIONS HAVE BEEN COMPUTED
            if st.session_state.get('show_results', False) and 'predictions_df' in st.session_state:
                results_df = st.session_state.predictions_df

                st.success("✅ **This uploaded data is now the PRIMARY data source for all pages!**")
                st.info("Navigate to Dashboard, Students List, or Analytics to see analysis of YOUR uploaded data.")

                st.markdown("### 📊 RESULTS")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("TOTAL", len(results_df))
                with col2:
                    high_risk = len(results_df[results_df['risk_level'] == 'High'])
                    st.metric("HIGH RISK", high_risk)
                with col3:
                    medium_risk = len(results_df[results_df['risk_level'] == 'Medium'])
                    st.metric("MEDIUM RISK", medium_risk)
                with col4:
                    low_risk = len(results_df[results_df['risk_level'] == 'Low'])
                    st.metric("LOW RISK", low_risk)

                st.markdown("### DETAILED RESULTS")
                st.dataframe(results_df, use_container_width=True)

                # HIGH RISK STUDENTS SECTION with notification
                high_risk_students = st.session_state.get('high_risk_students', results_df[results_df['risk_level'] == 'High'].copy())

                if len(high_risk_students) > 0:
                    st.markdown("### 🚨 HIGH RISK STUDENTS - IMMEDIATE ATTENTION REQUIRED")

                    # Display high-risk students in a highlighted box
                    st.markdown("""
                    <div style='background-color: #FFE4E4; border: 5px solid #FF0000;
                                padding: 20px; box-shadow: 8px 8px 0px #000; margin: 20px 0;'>
                        <h3 style='color: #FF0000; text-transform: uppercase; margin: 0;'>
                            ⚠️ {count} Students Need Urgent Intervention
                        </h3>
                    </div>
                    """.format(count=len(high_risk_students)), unsafe_allow_html=True)

                    # Display the high-risk students table - ALWAYS show student_id, dropout_probability, risk_level, email
                    display_cols = ['student_id', 'dropout_probability', 'risk_level', 'email']

                    st.dataframe(
                        high_risk_students[display_cols].style.background_gradient(
                            subset=['dropout_probability'], cmap='Reds'
                        ),
                        use_container_width=True
                    )

                    # Notify button with n8n webhook integration
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.info("📧 Click the button to send email notifications to all high-risk students via n8n workflow")

                    with col2:
                        if st.button("🔔 NOTIFY THEM", use_container_width=True, type="primary", key="notify_them_btn"):
                            # Check if email column exists and has valid emails
                            has_valid_emails = 'email' in high_risk_students.columns and high_risk_students['email'].notna().any() and (high_risk_students['email'] != 'N/A').any()

                            if has_valid_emails:
                                with st.spinner("Sending notifications via n8n..."):
                                    try:
                                        # n8n webhook URL (can be overridden via Streamlit secrets N8N_WEBHOOK_URL)
                                        try:
                                            n8n_webhook_url = st.secrets["N8N_WEBHOOK_URL"]
                                        except:
                                            n8n_webhook_url = "https://wwer45367iujhgf.app.n8n.cloud/webhook-test/cd29ebfe-f101-4155-9426-28527d0e5090"

                                        # Prepare payload for n8n - only include students with valid emails
                                        students_data = []
                                        for _, student in high_risk_students.iterrows():
                                            email = student.get('email', '')
                                            # Only include if email exists, is not NaN, and is not 'N/A'
                                            if pd.notna(email) and email != 'N/A' and email != '' and '@' in str(email):
                                                students_data.append({
                                                    'student_id': str(student['student_id']),
                                                    'email': str(email),
                                                    'dropout_probability': float(student['dropout_probability']),
                                                    'risk_level': student['risk_level'],
                                                    'prediction': student.get('prediction', 'Dropout')
                                                })

                                        # Check if we have any valid emails to send
                                        if len(students_data) == 0:
                                            st.warning("⚠️ No valid email addresses found in high-risk students. Please ensure your CSV has valid email addresses.")
                                        else:
                                            payload = {
                                                'total_high_risk': len(high_risk_students),
                                                'students_with_emails': len(students_data),
                                                'students': students_data,
                                                'timestamp': pd.Timestamp.now().isoformat(),
                                                'source': 'Student Dropout Prediction System'
                                            }

                                            # Send to n8n webhook
                                            response = requests.post(
                                                n8n_webhook_url,
                                                json=payload,
                                                timeout=30
                                            )

                                            if response.status_code == 200:
                                                st.success(f"✅ Successfully sent notifications to {len(students_data)} students via n8n workflow!")
                                                st.balloons()
                                            else:
                                                st.error(f"❌ Failed to send notifications. Status: {response.status_code}")
                                                st.info("💡 Please check your n8n webhook URL in Streamlit secrets")

                                    except requests.exceptions.Timeout:
                                        st.error("⏱️ Request timed out. Please check your n8n instance.")
                                    except Exception as e:
                                        st.error(f"❌ Error sending notifications: {str(e)}")
                                        st.info("""
                                        **To set up n8n webhook:**
                                        1. Create a workflow in n8n
                                        2. Add a Webhook trigger node
                                        3. Add email nodes to send notifications
                                        4. Copy webhook URL
                                        5. Add to Streamlit secrets as N8N_WEBHOOK_URL

                                        Or update the URL directly in app.py
                                        """)
                            else:
                                st.warning("⚠️ No email addresses found in the uploaded data. Please include an 'email' column in your CSV.")
                else:
                    st.success("✅ No high-risk students found in this batch!")

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="DOWNLOAD RESULTS CSV",
                    data=csv,
                    file_name="dropout_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Students List":
    st.markdown("# 👥 STUDENTS LIST")
    st.markdown("### Browse and search students")

    # Data source indicator
    if st.session_state.data_source == "uploaded":
        st.info("📊 **Showing students from YOUR uploaded data!**")
    else:
        st.info("📁 **Showing students from default dataset.** Upload CSV in 'Predict Batch' to view your own students.")

    # Search and filters
    col1, col2, col3 = st.columns(3)

    with col1:
        search_id = st.text_input("SEARCH BY ID")

    with col2:
        dept_filter = st.selectbox("FILTER BY DEPARTMENT", ["All"] + list(df['department'].unique()))

    with col3:
        risk_filter = st.selectbox("FILTER BY RISK", ["All", "High", "Medium", "Low"])

    # Prepare predictions - remove email column if it exists (email is only for notifications)
    cols_to_drop = ['student_id', 'dropout']
    if 'email' in df.columns:
        cols_to_drop.append('email')

    X = df.drop(cols_to_drop, axis=1)
    for col in label_encoders.keys():
        if col in X.columns:
            le = label_encoders[col]
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col])

    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df_display = df.copy()
    df_display['dropout_probability'] = probabilities
    df_display['risk_level'] = df_display['dropout_probability'].apply(
        lambda x: 'High' if x >= 0.6 else ('Medium' if x >= 0.3 else 'Low')
    )

    # Apply filters
    if search_id:
        df_display = df_display[df_display['student_id'].astype(str).str.contains(search_id)]

    if dept_filter != "All":
        df_display = df_display[df_display['department'] == dept_filter]

    if risk_filter != "All":
        df_display = df_display[df_display['risk_level'] == risk_filter]

    st.markdown(f"### SHOWING {len(df_display)} STUDENTS")

    # Display table
    display_cols = ['student_id', 'gender', 'department', 'cgpa', 'attendance_rate',
                    'dropout_probability', 'risk_level', 'dropout']
    st.dataframe(
        df_display[display_cols].head(100),
        use_container_width=True,
        column_config={
            "student_id": "ID",
            "gender": "Gender",
            "department": "Department",
            "cgpa": st.column_config.NumberColumn("CGPA", format="%.2f"),
            "attendance_rate": st.column_config.NumberColumn("Attendance", format="%.1f%%"),
            "dropout_probability": st.column_config.ProgressColumn(
                "Dropout Risk",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "risk_level": "Risk Level",
            "dropout": st.column_config.CheckboxColumn("Dropout")
        }
    )

    # Student detail view
    st.markdown("---")
    st.markdown("### 🔍 STUDENT DETAIL VIEW")
    selected_id = st.number_input("ENTER STUDENT ID", min_value=int(df['student_id'].min()),
                                   max_value=int(df['student_id'].max()))

    if st.button("VIEW DETAILS"):
        student = df[df['student_id'] == selected_id]
        if not student.empty:
            student_data = student.iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### PERSONAL INFO")
                st.write(f"**ID:** {student_data['student_id']}")
                st.write(f"**Gender:** {student_data['gender']}")
                st.write(f"**Age:** {student_data['age']}")
                st.write(f"**Department:** {student_data['department']}")
                st.write(f"**Scholarship:** {student_data['scholarship']}")

                st.markdown("#### ACADEMIC INFO")
                st.write(f"**CGPA:** {student_data['cgpa']}")
                st.write(f"**Attendance:** {student_data['attendance_rate']}%")
                st.write(f"**Past Failures:** {student_data['past_failures']}")
                st.write(f"**Assignments Submitted:** {student_data['assignments_submitted']}")
                st.write(f"**Projects Completed:** {student_data['projects_completed']}")

            with col2:
                st.markdown("#### ACTIVITIES & BACKGROUND")
                st.write(f"**Extra Curricular:** {student_data['extra_curricular']}")
                st.write(f"**Sports Participation:** {student_data['sports_participation']}")
                st.write(f"**Total Activities:** {student_data['total_activities']}")
                st.write(f"**Study Hours/Week:** {student_data['study_hours_per_week']}")
                st.write(f"**Family Income:** ₹{student_data['family_income']:,}")
                st.write(f"**Parental Education:** {student_data['parental_education']}")

                st.markdown("#### DROPOUT STATUS")
                actual_dropout = "Yes" if student_data['dropout'] == 1 else "No"
                st.write(f"**Actual Dropout:** {actual_dropout}")

                # Prediction
                student_dict = student_data.to_dict()
                features = {k: v for k, v in student_dict.items() if k in FEATURE_COLUMNS}
                prediction, probability = predict_dropout(features)
                dropout_prob = probability[1] * 100
                risk_level = "HIGH" if dropout_prob >= 60 else ("MEDIUM" if dropout_prob >= 30 else "LOW")

                st.write(f"**Predicted Risk:** {dropout_prob:.1f}%")
                st.write(f"**Risk Level:** {risk_level}")
        else:
            st.error("Student ID not found")

elif page == "Analytics":
    st.markdown("# 📊 ADVANCED ANALYTICS")
    st.markdown("### Deep dive into dropout patterns")

    # Data source indicator
    if st.session_state.data_source == "uploaded":
        st.info("📊 **Analyzing YOUR uploaded data!**")
    else:
        st.info("📁 **Analyzing default dataset.** Upload CSV in 'Predict Batch' to analyze your own data.")

    # Prepare predictions for all students - remove email column if it exists (email is only for notifications)
    cols_to_drop = ['student_id', 'dropout']
    if 'email' in df.columns:
        cols_to_drop.append('email')

    X = df.drop(cols_to_drop, axis=1)
    for col in label_encoders.keys():
        if col in X.columns:
            le = label_encoders[col]
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col])

    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df_analytics = df.copy()
    df_analytics['dropout_probability'] = probabilities
    df_analytics['risk_level'] = df_analytics['dropout_probability'].apply(
        lambda x: 'High' if x >= 0.6 else ('Medium' if x >= 0.3 else 'Low')
    )

    # Risk distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### RISK DISTRIBUTION")
        risk_counts = df_analytics['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    color=risk_counts.index,
                    color_discrete_map={'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#A8E6CF'})
        fig.update_layout(
            paper_bgcolor='white',
            font=dict(family="Space Grotesk", color="black", weight=700)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### DEPARTMENT RISK ANALYSIS")
        dept_risk = df_analytics.groupby('department')['dropout_probability'].mean() * 100
        fig = px.bar(dept_risk.reset_index(), x='department', y='dropout_probability',
                    color='dropout_probability', color_continuous_scale='Reds',
                    labels={'dropout_probability': 'Avg Dropout Risk (%)'})
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Space Grotesk", color="black", weight=700)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("### CORRELATION ANALYSIS")
    numerical_cols = ['age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures',
                     'study_hours_per_week', 'assignments_submitted', 'projects_completed',
                     'total_activities', 'dropout']
    corr_matrix = df[numerical_cols].corr()

    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=numerical_cols,
                    y=numerical_cols,
                    color_continuous_scale='RdBu_r',
                    aspect="auto")
    fig.update_layout(
        paper_bgcolor='white',
        font=dict(family="Space Grotesk", color="black", weight=700)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.markdown("### FEATURE DISTRIBUTIONS")

    col1, col2 = st.columns(2)

    with col1:
        feature = st.selectbox("SELECT FEATURE", numerical_cols[:-1])

    with col2:
        chart_type = st.selectbox("CHART TYPE", ["Histogram", "Box Plot", "Violin Plot"])

    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color='dropout',
                          color_discrete_map={0: '#A8E6CF', 1: '#FF6B6B'},
                          barmode='overlay')
    elif chart_type == "Box Plot":
        fig = px.box(df, x='dropout', y=feature,
                    color='dropout',
                    color_discrete_map={0: '#A8E6CF', 1: '#FF6B6B'})
    else:
        fig = px.violin(df, x='dropout', y=feature,
                       color='dropout',
                       color_discrete_map={0: '#A8E6CF', 1: '#FF6B6B'})

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Space Grotesk", color="black", weight=700)
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Simulator":
    st.markdown("# 🎮 PREDICTION SIMULATOR")
    st.markdown("### Test different scenarios")

    st.info("Adjust the sliders to see how different factors affect dropout probability")

    col1, col2 = st.columns(2)

    with col1:
        sim_cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
        sim_attendance = st.slider("ATTENDANCE RATE (%)", 0.0, 100.0, 75.0, 1.0)
        sim_study_hours = st.slider("STUDY HOURS/WEEK", 0.0, 60.0, 20.0, 1.0)
        sim_assignments = st.slider("ASSIGNMENTS SUBMITTED", 0, 100, 40, 1)
        sim_projects = st.slider("PROJECTS COMPLETED", 0, 10, 3, 1)

    with col2:
        sim_activities = st.slider("TOTAL ACTIVITIES", 0, 20, 5, 1)
        sim_past_failures = st.slider("PAST FAILURES", 0, 10, 0, 1)
        sim_age = st.slider("AGE", 16, 30, 20, 1)
        sim_income = st.slider("FAMILY INCOME", 0, 200000, 50000, 5000)

        sim_gender = st.selectbox("GENDER", df['gender'].unique(), key='sim_gender')
        sim_dept = st.selectbox("DEPARTMENT", df['department'].unique(), key='sim_dept')
        sim_scholarship = st.selectbox("SCHOLARSHIP", df['scholarship'].unique(), key='sim_scholarship')
        sim_parental_edu = st.selectbox("PARENTAL EDUCATION", df['parental_education'].unique(), key='sim_parental')
        sim_extra = st.selectbox("EXTRA CURRICULAR", df['extra_curricular'].unique(), key='sim_extra')
        sim_sports = st.selectbox("SPORTS PARTICIPATION", df['sports_participation'].unique(), key='sim_sports')

    if st.button("RUN SIMULATION", use_container_width=True):
        sim_data = {
            'gender': sim_gender,
            'department': sim_dept,
            'scholarship': sim_scholarship,
            'parental_education': sim_parental_edu,
            'extra_curricular': sim_extra,
            'age': sim_age,
            'cgpa': sim_cgpa,
            'attendance_rate': sim_attendance,
            'family_income': sim_income,
            'past_failures': sim_past_failures,
            'study_hours_per_week': sim_study_hours,
            'assignments_submitted': sim_assignments,
            'projects_completed': sim_projects,
            'total_activities': sim_activities,
            'sports_participation': sim_sports
        }

        prediction, probability = predict_dropout(sim_data)
        dropout_prob = probability[1] * 100

        st.markdown("---")
        st.markdown("## SIMULATION RESULTS")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("DROPOUT PROBABILITY", f"{dropout_prob:.1f}%")

        with col2:
            risk_level = "HIGH" if dropout_prob >= 60 else ("MEDIUM" if dropout_prob >= 30 else "LOW")
            st.metric("RISK LEVEL", risk_level)

        with col3:
            status = "AT RISK" if prediction == 1 else "SAFE"
            st.metric("STATUS", status)

        st.progress(dropout_prob / 100)

        # What-if scenarios
        st.markdown("### WHAT-IF SCENARIOS")

        scenarios = []

        # Improve CGPA
        if sim_cgpa < 8.0:
            improved_data = sim_data.copy()
            improved_data['cgpa'] = min(sim_cgpa + 1.0, 10.0)
            _, prob = predict_dropout(improved_data)
            scenarios.append({
                'scenario': 'Improve CGPA by 1.0',
                'current': f"{dropout_prob:.1f}%",
                'predicted': f"{prob[1]*100:.1f}%",
                'change': f"{(prob[1]*100 - dropout_prob):.1f}%"
            })

        # Improve attendance
        if sim_attendance < 90.0:
            improved_data = sim_data.copy()
            improved_data['attendance_rate'] = min(sim_attendance + 10.0, 100.0)
            _, prob = predict_dropout(improved_data)
            scenarios.append({
                'scenario': 'Increase attendance by 10%',
                'current': f"{dropout_prob:.1f}%",
                'predicted': f"{prob[1]*100:.1f}%",
                'change': f"{(prob[1]*100 - dropout_prob):.1f}%"
            })

        # Increase study hours
        if sim_study_hours < 30:
            improved_data = sim_data.copy()
            improved_data['study_hours_per_week'] = min(sim_study_hours + 10.0, 60.0)
            _, prob = predict_dropout(improved_data)
            scenarios.append({
                'scenario': 'Study 10 more hours/week',
                'current': f"{dropout_prob:.1f}%",
                'predicted': f"{prob[1]*100:.1f}%",
                'change': f"{(prob[1]*100 - dropout_prob):.1f}%"
            })

        # More activities
        if sim_activities < 10:
            improved_data = sim_data.copy()
            improved_data['total_activities'] = min(sim_activities + 3, 20)
            _, prob = predict_dropout(improved_data)
            scenarios.append({
                'scenario': 'Join 3 more activities',
                'current': f"{dropout_prob:.1f}%",
                'predicted': f"{prob[1]*100:.1f}%",
                'change': f"{(prob[1]*100 - dropout_prob):.1f}%"
            })

        if scenarios:
            scenarios_df = pd.DataFrame(scenarios)
            st.dataframe(scenarios_df, use_container_width=True, hide_index=True)


# ================================
# FLOATING CHATBOT - APPEARS ON ALL PAGES
# ================================

# Initialize chatbot state
if 'chatbot_open' not in st.session_state:
    st.session_state.chatbot_open = False

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI guidance assistant. I can help you with:\n\n- Understanding dropout risk factors\n- Suggesting interventions for at-risk students\n- Explaining prediction results\n- Providing academic resources and guidance\n\nHow can I help you today?"}
    ]

# Create a fixed bottom-right chatbot toggle button
col_spacer, col_button = st.columns([5, 1])
with col_button:
    if st.button("💬 Chat", key="chatbot_toggle", help="Open AI Assistant", use_container_width=True):
        st.session_state.chatbot_open = not st.session_state.chatbot_open
        st.rerun()

# Display chatbot when open
if st.session_state.chatbot_open:
    st.markdown("---")
    st.markdown("### 🤖 AI GUIDANCE ASSISTANT")

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("✖ Close", key="close_chat"):
            st.session_state.chatbot_open = False
            st.rerun()

    # Display chat messages in an expander or container
    chat_container = st.container()

    with chat_container:
        # Display chat messages
        for message in st.session_state.messages[-5:]:  # Show last 5 messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about student dropout prediction..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate response using OpenAI
            if OPENAI_ENABLED:
                try:
                    # Build dynamic context from ALL pages data
                    dynamic_context = f"""
## CURRENT SESSION DATA:

**Current Page:** {page}
**Data Source:** {"Uploaded CSV Data" if st.session_state.data_source == "uploaded" else "Default Training Dataset"}
**Total Students:** {len(df):,}
**Dropout Count:** {df['dropout'].sum():,}
**Dropout Rate:** {(df['dropout'].sum() / len(df)) * 100:.2f}%
**Average CGPA:** {df['cgpa'].mean():.2f}
**Average Attendance:** {df['attendance_rate'].mean():.2f}%
"""

                    # Add predictions data if available
                    if 'predictions_df' in st.session_state and st.session_state.predictions_df is not None:
                        pred_df = st.session_state.predictions_df
                        high_risk_count = len(pred_df[pred_df['risk_level'] == 'High'])
                        medium_risk_count = len(pred_df[pred_df['risk_level'] == 'Medium'])
                        low_risk_count = len(pred_df[pred_df['risk_level'] == 'Low'])

                        dynamic_context += f"""
**PREDICTION RESULTS:**
- Total Predictions: {len(pred_df)}
- High Risk Students: {high_risk_count}
- Medium Risk Students: {medium_risk_count}
- Low Risk Students: {low_risk_count}
"""

                        # Add high-risk students details if available
                        if 'high_risk_students' in st.session_state and len(st.session_state.high_risk_students) > 0:
                            dynamic_context += f"\n**HIGH RISK STUDENTS (need immediate attention):** {len(st.session_state.high_risk_students)}\n"

                    # Add analysis results if available
                    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
                        results = st.session_state.analysis_results
                        dynamic_context += f"""
**ANALYSIS SUMMARY:**
- Total Students Analyzed: {results.get('total', 0)}
- High Risk Students: {results.get('high_risk', 0)}
- Medium Risk Students: {results.get('medium_risk', 0)}
- Low Risk Students: {results.get('low_risk', 0)}
- Predicted Dropout Rate: {results.get('dropout_rate', 0):.2f}%
"""

                    # Department breakdown
                    if 'department' in df.columns:
                        dept_stats = df.groupby('department')['dropout'].agg(['sum', 'count'])
                        dept_stats['rate'] = (dept_stats['sum'] / dept_stats['count']) * 100
                        dynamic_context += "\n**DEPARTMENT BREAKDOWN:**\n"
                        for dept, row in dept_stats.iterrows():
                            dynamic_context += f"- {dept}: {row['rate']:.1f}% dropout rate ({int(row['sum'])}/{int(row['count'])} students)\n"

                    # Prepare conversation history for OpenAI
                    messages_for_api = [
                        {"role": "system", "content": SYSTEM_CONTEXT + "\n\n" + dynamic_context}
                    ]

                    # Add recent conversation history (last 10 messages for context)
                    recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
                    for msg in recent_messages:
                        messages_for_api.append({"role": msg["role"], "content": msg["content"]})

                    # Call OpenAI API
                    with st.spinner("Thinking..."):
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages_for_api,
                            temperature=0.7,
                            max_tokens=1000
                        )

                    response = completion.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

                except Exception as e:
                    error_response = f"I apologize, but I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
                    st.session_state.messages.append({"role": "assistant", "content": error_response})
                    st.rerun()

            else:
                # Fallback to basic response if OpenAI unavailable
                response = "I can help you with understanding dropout risk factors, intervention strategies, and academic guidance. (Note: Full AI capabilities require OpenAI integration)"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()



# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
<strong>STUDENT DROPOUT PREDICTOR</strong><br>
AI-Powered Early Warning System<br>
The National College, Basavanagudi | Tech Symposium Hackathon
</div>
""", unsafe_allow_html=True)
