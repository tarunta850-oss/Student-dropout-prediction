# 🎓 STUDENT DROPOUT PREDICTION SYSTEM

**AI-Powered Early Warning System for Higher Education**

An intelligent, data-driven solution to predict student dropout risk and enable timely interventions. Built for **The National College, Basavanagudi** Tech Symposium Hackathon.

---

## 🚀 PROJECT OVERVIEW

This project addresses the critical challenge of student dropout in higher education institutions across India. With nearly **25% of students struggling to complete their degrees on time**, our ML-powered system provides:

- **Early Detection**: Identify at-risk students before performance deteriorates
- **Predictive Analytics**: ML models trained on 19,591+ student records
- **Actionable Insights**: Risk scores, recommendations, and intervention strategies
- **Interactive Dashboard**: Neo-Brutalism styled UI for easy data visualization
- **AI Chatbot**: Personalized guidance and support recommendations

---

## 📊 DATASET

**19,591 student records** with **17 features**:

### Features:
- **Demographics**: gender, age, department
- **Academic**: cgpa, attendance_rate, past_failures, assignments_submitted, projects_completed
- **Behavioral**: study_hours_per_week, extra_curricular, sports_participation, total_activities
- **Socio-Economic**: family_income, parental_education, scholarship
- **Target**: dropout (0 = No Dropout, 1 = Dropout)

### Statistics:
- **Total Students**: 19,591
- **Dropout Rate**: 28.2%
- **Features**: 15 input features
- **No Missing Values**: Clean dataset

---

## 🤖 MACHINE LEARNING MODELS

### Models Trained:
1. **Logistic Regression** ⭐ (Best Model)
2. Gradient Boosting
3. AdaBoost
4. Random Forest
5. XGBoost
6. Decision Tree

### Best Model Performance:
- **Model**: Logistic Regression
- **Accuracy**: 73.82%
- **Precision**: 58.11%
- **Recall**: 25.61%
- **F1 Score**: 35.55%
- **ROC AUC**: **0.7305** 🎯

### Risk Classification:
- **Low Risk**: < 30% dropout probability
- **Medium Risk**: 30-60% dropout probability
- **High Risk**: > 60% dropout probability

---

## 🎨 FEATURES

### ✅ Core Features:
1. **Dashboard** - KPIs, statistics, and visual analytics
2. **Single Student Prediction** - Detailed risk assessment for individual students
3. **Batch Prediction** - CSV upload for multiple students
4. **Students List** - Browse, search, and filter students by risk level
5. **Advanced Analytics** - Correlation analysis, department-wise breakdown, feature distributions
6. **Prediction Simulator** - Interactive what-if scenario testing
7. **AI Chatbot** - Intelligent guidance and recommendations

### 🌟 Bonus Features:
- **SHAP Explainability** - Feature importance visualization
- **CSV Export** - Download prediction results
- **Interactive Charts** - Plotly-powered visualizations
- **Real-time Predictions** - Instant risk assessment
- **Neo-Brutalism UI** - Modern, colorful, and bold design

---

## 🛠️ TECHNOLOGY STACK

### Backend:
- **Flask** - REST API server
- **scikit-learn** - ML model training
- **XGBoost** - Gradient boosting
- **SHAP** - Model explainability
- **Pandas/NumPy** - Data processing

### Frontend:
- **Streamlit** - Interactive web dashboard
- **Plotly** - Data visualizations
- **Custom CSS** - Neo-Brutalism design

### ML Pipeline:
- Data preprocessing
- Label encoding
- Standard scaling
- Multiple model training
- Cross-validation
- Model persistence (joblib)

---

## 📁 PROJECT STRUCTURE

```
Natinal_HAK/
│
├── dataset.xlsx                      # Original dataset
├── cleaned_dataset.csv               # Processed dataset
├── Tech Symposium Hackathon.pdf      # Problem statement
│
├── data_preprocessing.py             # Data cleaning & EDA
├── train_models.py                   # ML model training
├── shap_analysis.py                  # SHAP explainability
├── api.py                            # Flask REST API
├── app.py                            # Streamlit dashboard
│
├── models/                           # Trained models directory
│   ├── best_model.pkl               # Best performing model
│   ├── scaler.pkl                   # Feature scaler
│   ├── label_encoders.pkl           # Categorical encoders
│   ├── feature_importance.csv       # Feature importance
│   ├── model_comparison.csv         # Model metrics
│   ├── shap_values.npy             # SHAP values
│   ├── shap_summary_plot.png       # SHAP visualizations
│   └── shap_bar_plot.png           # SHAP bar chart
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## ⚡ INSTALLATION & SETUP

### Prerequisites:
- Python 3.8+
- pip package manager

### Step 1: Clone/Download Project
```bash
cd Natinal_HAK
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Data Preprocessing (Optional - already done)
```bash
python data_preprocessing.py
```

### Step 4: Train Models (Optional - already done)
```bash
python train_models.py
```

### Step 5: Generate SHAP Analysis (Optional)
```bash
python shap_analysis.py
```

---

## 🚀 RUNNING THE APPLICATION

### Option 1: Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
- Access at: `http://localhost:8501`
- Features full Neo-Brutalism UI
- All features integrated

### Option 2: Flask API
```bash
python api.py
```
- API runs at: `http://localhost:5000`
- RESTful endpoints for predictions

---

## 🔌 API ENDPOINTS

### Base URL: `http://localhost:5000`

#### 1. Home
```
GET /
Returns: API status and version
```

#### 2. Single Prediction
```
POST /api/predict
Body: {
  "gender": "MALE",
  "department": "Engineering",
  "cgpa": 7.5,
  "attendance_rate": 75.0,
  ...
}
Returns: Dropout probability and risk level
```

#### 3. Batch Prediction
```
POST /api/predict-batch
Body: CSV file upload
Returns: Predictions for all students
```

#### 4. Get All Students
```
GET /api/students?page=1&per_page=50&risk=all
Returns: Paginated student list with predictions
```

#### 5. Get Student Details
```
GET /api/student/<student_id>
Returns: Detailed student information and prediction
```

#### 6. Get Statistics
```
GET /api/statistics
Returns: Overall analytics and statistics
```

#### 7. Feature Importance
```
GET /api/feature-importance
Returns: Feature importance scores
```

---

## 📱 USER INTERFACE

### Neo-Brutalism Design Theme:
- **Bold Colors**: Bright, high-contrast color palette
- **Thick Borders**: 3-4px solid black borders
- **Box Shadows**: Offset shadows for depth
- **Sharp Corners**: No border radius
- **Colorful Badges**: Risk level indicators
- **Space Grotesk Font**: Modern, clean typography

### Color Palette:
- **Primary Yellow**: #FFE66D
- **Danger Red**: #FF6B6B
- **Success Green**: #A8E6CF
- **Warning Yellow**: #FFD93D
- **Info Cyan**: #4ECDC4
- **Background**: Gradient from #FFF5E1 to #FFE4E1

---

## 🎯 USE CASES

### For Administrators:
- Monitor overall dropout trends
- Identify departments needing support
- Allocate resources effectively
- Track intervention success rates

### For Counselors:
- Prioritize at-risk students
- Personalized intervention planning
- Track student progress over time
- Access recommended actions

### For Faculty:
- Early warning for struggling students
- Data-driven mentoring
- Understand risk factors
- Proactive engagement

---

## 📈 MODEL INSIGHTS

### Top Risk Factors (Most Important Features):
1. **CGPA** - Strong negative correlation with dropout
2. **Attendance Rate** - Critical indicator
3. **Past Failures** - Historical performance matters
4. **Study Hours** - Time investment pays off
5. **Assignments Submitted** - Engagement indicator
6. **Family Income** - Socio-economic factor
7. **Extra Curricular** - Campus engagement
8. **Projects Completed** - Hands-on learning
9. **Parental Education** - Background influence
10. **Total Activities** - Social integration

### Key Findings:
- Students with CGPA < 6.0 have **3x higher** dropout risk
- Attendance < 70% is a **critical warning sign**
- Scholarship holders have **lower dropout rates**
- Extra-curricular participation **reduces risk by 40%**
- Sports participation shows **strong protective effect**

---

## 🤝 INTERVENTION STRATEGIES

### High Risk (>60%):
- ⚠️ **Immediate counseling session**
- 📞 **Contact parents/guardians**
- 📝 **Personalized academic support plan**
- 📊 **Daily attendance monitoring**
- 👥 **Peer mentorship assignment**
- 💰 **Financial aid review**

### Medium Risk (30-60%):
- 📅 **Bi-weekly check-ins**
- 📚 **Study skills workshop**
- ⏰ **Time management training**
- 🎯 **Goal setting sessions**
- 🤝 **Connect with academic advisors**

### Low Risk (<30%):
- ✅ **Monthly monitoring**
- 🌟 **Recognition programs**
- 💪 **Leadership opportunities**
- 🎓 **Maintain engagement**

---

## 🧪 TESTING

### Test the Application:

1. **Dashboard**: View overall statistics and trends
2. **Single Prediction**: Enter custom student data
3. **Batch Upload**: Use the cleaned_dataset.csv
4. **Student Search**: Search by ID (200000-219999)
5. **Simulator**: Adjust sliders to test scenarios
6. **Chatbot**: Ask questions about dropout factors

### Sample Test Data:
```python
{
  "gender": "MALE",
  "department": "Engineering",
  "scholarship": "Yes",
  "parental_education": "Graduate",
  "extra_curricular": "Yes",
  "age": 20,
  "cgpa": 6.5,
  "attendance_rate": 72.0,
  "family_income": 45000,
  "past_failures": 1,
  "study_hours_per_week": 18.0,
  "assignments_submitted": 35,
  "projects_completed": 2,
  "total_activities": 3,
  "sports_participation": "NO"
}
```

---

## 📊 DELIVERABLES CHECKLIST

### Required:
- ✅ Cleaned dataset (CSV)
- ✅ Trained ML model (.pkl)
- ✅ Frontend dashboard (Streamlit)
- ✅ README documentation
- ✅ Workflow explanation

### Bonus:
- ✅ Feature importance visualization (SHAP)
- ✅ CSV upload for batch predictions
- ✅ Interactive analytics dashboard
- ✅ API integration (Flask)
- ✅ Explainable AI (SHAP)
- ✅ Chatbot for guidance

---

## 🎓 ACADEMIC INTEGRITY

This project was developed for **The National College, Basavanagudi** Tech Symposium Hackathon to address real-world challenges in higher education.

### Problem Statement:
- Predict student dropout risk early
- Enable timely interventions
- Support institutional decision-making
- Create responsive educational environment

---

## 🔮 FUTURE ENHANCEMENTS

- Integration with LMS systems
- Real-time data pipeline
- Mobile application
- SMS/Email alert system
- Advanced NLP chatbot with GPT
- Temporal analysis (semester-wise tracking)
- Multi-institutional deployment
- Parent portal access
- Intervention tracking system

---

## 🐛 TROUBLESHOOTING

### Common Issues:

**1. Models not loading:**
```bash
# Re-train models
python train_models.py
```

**2. Port already in use:**
```bash
# Change port in api.py or app.py
# For Streamlit: streamlit run app.py --server.port 8502
```

**3. Module not found:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**4. CSV upload fails:**
- Ensure CSV has all required columns
- Check data types match expected format
- Remove any special characters

---

## 📞 SUPPORT

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with sample data provided

---

## 🏆 HACKATHON SUBMISSION

**Event**: Tech Symposium Hackathon
**Institution**: The National College, Basavanagudi (Autonomous)
**Location**: Bengaluru - 560004
**Theme**: AI for Education

### Submission Includes:
1. Complete source code
2. Trained ML models
3. Interactive dashboard
4. API endpoints
5. Comprehensive documentation
6. SHAP explainability
7. Batch prediction feature
8. AI chatbot integration

---

## 🎉 ACKNOWLEDGMENTS

- **The National College, Basavanagudi** for organizing the hackathon
- **scikit-learn** for ML framework
- **Streamlit** for rapid dashboard development
- **SHAP** for model interpretability
- **Plotly** for beautiful visualizations

---

## 📜 LICENSE

This project is developed for educational and research purposes as part of a hackathon submission.

---

## 🚀 QUICK START GUIDE

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py

# 3. Open browser at http://localhost:8501

# 4. Explore features:
#    - Dashboard (overview)
#    - Predict Single (test predictions)
#    - Students List (browse data)
#    - Analytics (deep dive)
#    - Simulator (what-if scenarios)
#    - Chatbot (get guidance)
```

---

## 💡 KEY HIGHLIGHTS

- 📊 **19,591 student records** analyzed
- 🤖 **6 ML models** trained and compared
- 🎯 **73.8% accuracy** achieved
- 🎨 **Neo-Brutalism UI** with colorful, creative design
- ⚡ **Real-time predictions** in seconds
- 📈 **Interactive visualizations** with Plotly
- 🔍 **SHAP explainability** for transparency
- 💬 **AI chatbot** for guidance
- 📤 **Batch processing** for efficiency
- 🌐 **REST API** for integration

---

**JAI SHREE RAM** 🙏

Built with ❤️ for student success and educational excellence.

---

*For demo or questions, refer to the code comments and inline documentation.*
