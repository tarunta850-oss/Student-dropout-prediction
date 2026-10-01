import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# Page configuration
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
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

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #4ECDC4 !important;
        border-right: 5px solid #000 !important;
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
        background: rgba(255,255,255,0.3) !important;
        padding: 10px !important;
        border: 2px solid #000 !important;
        margin: 5px 0 !important;
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

    /* Better text visibility */
    p, span, div, label {
        color: #000 !important;
    }

    /* Markdown text */
    .stMarkdown {
        color: #000 !important;
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        border: 3px solid #000 !important;
        border-radius: 0px !important;
        box-shadow: 5px 5px 0px #000 !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #FFF !important;
        border: 3px solid #000 !important;
        box-shadow: 5px 5px 0px #000 !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }

    /* Ensure all text is black */
    * {
        color: #000 !important;
    }

    /* White background for main content */
    .main .block-container {
        padding: 2rem !important;
    }
</style>
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
    df = load_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

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

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='color: #000 !important; text-shadow: 3px 3px 0px #FFE66D;'>🎓 NAVIGATION</h1>", unsafe_allow_html=True)
    page = st.radio("SELECT PAGE:",
                    ["Dashboard", "Predict Single", "Predict Batch", "Students List",
                     "Analytics", "Simulator", "Chatbot"],
                    label_visibility="visible")

    st.markdown("---")
    st.markdown("<h3 style='color: #000 !important;'>📊 QUICK STATS</h3>", unsafe_allow_html=True)
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
    """)

    uploaded_file = st.file_uploader("UPLOAD CSV FILE", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} students")

            st.markdown("### PREVIEW DATA")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("RUN PREDICTIONS", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    results = []

                    for idx, row in batch_df.iterrows():
                        try:
                            row_dict = row.to_dict()
                            prediction, probability = predict_dropout(row_dict)
                            dropout_prob = probability[1]

                            risk_level = "High" if dropout_prob >= 0.6 else ("Medium" if dropout_prob >= 0.3 else "Low")

                            results.append({
                                'student_id': row.get('student_id', idx + 1),
                                'prediction': 'Dropout' if prediction == 1 else 'No Dropout',
                                'dropout_probability': dropout_prob,
                                'risk_level': risk_level
                            })
                        except Exception as e:
                            results.append({
                                'student_id': row.get('student_id', idx + 1),
                                'error': str(e)
                            })

                    results_df = pd.DataFrame(results)

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

    # Search and filters
    col1, col2, col3 = st.columns(3)

    with col1:
        search_id = st.text_input("SEARCH BY ID")

    with col2:
        dept_filter = st.selectbox("FILTER BY DEPARTMENT", ["All"] + list(df['department'].unique()))

    with col3:
        risk_filter = st.selectbox("FILTER BY RISK", ["All", "High", "Medium", "Low"])

    # Prepare predictions
    X = df.drop(['student_id', 'dropout'], axis=1)
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

    # Prepare predictions for all students
    X = df.drop(['student_id', 'dropout'], axis=1)
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

elif page == "Chatbot":
    st.markdown("# 🤖 AI GUIDANCE CHATBOT")
    st.markdown("### Get personalized recommendations")

    st.info("Ask me questions about student dropout prediction, interventions, or academic guidance!")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI guidance assistant. I can help you with:\n\n- Understanding dropout risk factors\n- Suggesting interventions for at-risk students\n- Explaining prediction results\n- Providing academic resources and guidance\n\nHow can I help you today?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Simple rule-based responses
            prompt_lower = prompt.lower()

            if "dropout" in prompt_lower and ("why" in prompt_lower or "cause" in prompt_lower or "factor" in prompt_lower):
                response = """
**Key Dropout Risk Factors:**

1. **Academic Performance**
   - Low CGPA (< 6.0)
   - Poor attendance rate (< 70%)
   - Multiple past failures
   - Low assignment submission rate

2. **Behavioral Indicators**
   - Limited study hours
   - No participation in extra-curricular activities
   - No sports participation
   - Low project completion rate

3. **Socio-Economic Factors**
   - Low family income
   - Parents with limited education
   - No scholarship/financial support

4. **Personal Factors**
   - Age discrepancies
   - Academic stress
   - Lack of motivation
                """

            elif "intervention" in prompt_lower or "help" in prompt_lower or "support" in prompt_lower:
                response = """
**Intervention Strategies:**

**For High-Risk Students:**
- Immediate counseling session
- Contact parents/guardians
- Personalized academic support plan
- Daily attendance monitoring
- Mentorship program enrollment
- Financial aid assistance if needed
- Peer support group assignment

**For Medium-Risk Students:**
- Bi-weekly check-ins
- Study skills workshop
- Time management training
- Connect with academic advisors
- Encourage club participation

**For Low-Risk Students:**
- Monthly monitoring
- Maintain engagement
- Leadership opportunities
- Recognition programs
                """

            elif "cgpa" in prompt_lower or "grade" in prompt_lower:
                response = """
**CGPA Impact on Dropout:**

Students with CGPA below 6.0 have significantly higher dropout risk.

**Recommendations to Improve CGPA:**
- Regular tutoring sessions
- Study group participation
- Time management training
- Subject-specific support
- Test-taking strategies
- Assignment planning assistance
- Professor office hours attendance
                """

            elif "attendance" in prompt_lower:
                response = """
**Attendance & Dropout Relationship:**

Attendance rate below 70% is a strong predictor of dropout.

**Strategies to Improve Attendance:**
- Identify attendance barriers
- Flexible schedule options if needed
- Make classes more engaging
- Peer attendance buddies
- Reward good attendance
- Address health/transportation issues
- Parent communication
                """

            elif "scholarship" in prompt_lower or "financial" in prompt_lower or "money" in prompt_lower:
                response = """
**Financial Support Strategies:**

Financial stress is a major dropout factor.

**Available Support:**
- Merit-based scholarships
- Need-based grants
- Part-time campus jobs
- Emergency financial aid
- Payment plan options
- Book rental programs
- Transportation assistance
- Meal plan support

Recommend students contact Financial Aid office.
                """

            elif "predict" in prompt_lower or "model" in prompt_lower:
                response = """
**About the Prediction Model:**

Our ML model uses Logistic Regression trained on 19,591 student records.

**Model Performance:**
- Accuracy: 73.8%
- ROC AUC: 0.73

**Risk Levels:**
- **Low Risk** (<30%): Student is on track
- **Medium Risk** (30-60%): Needs attention
- **High Risk** (>60%): Immediate intervention required

The model analyzes 15 different features including academics, behavior, and socio-economic factors.
                """

            elif "activity" in prompt_lower or "activities" in prompt_lower:
                response = """
**Extra-Curricular Activities & Retention:**

Students engaged in activities show lower dropout rates.

**Benefits:**
- Improved sense of belonging
- Better time management
- Enhanced social connections
- Leadership development
- Stress relief

**Recommendations:**
- Encourage joining at least 2-3 activities
- Sports participation is highly beneficial
- Cultural clubs
- Technical societies
- Volunteer work
- Student government
                """

            elif "thanks" in prompt_lower or "thank you" in prompt_lower:
                response = "You're welcome! Feel free to ask if you have more questions about student support and dropout prevention."

            elif "hello" in prompt_lower or "hi" in prompt_lower:
                response = "Hello! How can I assist you with student dropout prediction and intervention strategies today?"

            else:
                response = """
I can help you with:
- **Understanding dropout risk factors**
- **Intervention strategies for at-risk students**
- **Academic performance improvement tips**
- **Financial support options**
- **Extra-curricular activity recommendations**
- **Model explanation and predictions**

Could you please ask a more specific question about any of these topics?
                """

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
<strong>STUDENT DROPOUT PREDICTOR</strong><br>
AI-Powered Early Warning System<br>
The National College, Basavanagudi | Tech Symposium Hackathon
</div>
""", unsafe_allow_html=True)
