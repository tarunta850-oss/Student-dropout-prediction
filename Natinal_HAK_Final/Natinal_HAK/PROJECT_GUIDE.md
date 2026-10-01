# 🎓 EduGuard AI: Student Dropout Prediction & Early Warning System
### VTU 5th Semester B.E. Mini Project (Computer Science & Engineering)

---

## 📌 1. Project Overview & Motivation
In higher education institutions, particularly under **Visvesvaraya Technological University (VTU)** and the **National Education Policy (NEP 2020)** framework, student retention and timely academic intervention are critical institutional priorities.

**EduGuard AI** is an intelligent early-warning decision-support system built using Machine Learning. It analyzes **15 multi-dimensional features** (academic metrics, behavioral indicators, and socio-economic factors) across **19,591 student records** to identify students at risk of dropout before semester completion.

---

## 🚀 2. System Architecture & Features

### 🌟 Key Capabilities
1. **Multi-Model Inference Engine**: Switch between 5 trained algorithms in real-time:
   - **Gradient Boosting** (Production Best: **73.23% Accuracy**, 71.65% ROC-AUC)
   - **Random Forest** (71.83% Accuracy, 71.86% ROC-AUC)
   - **XGBoost** (70.02% Accuracy, 70.95% ROC-AUC)
   - **Logistic Regression** (66.11% Accuracy, **68.05% High Sensitivity Recall**)
   - **Voting Ensemble** (71.17% Accuracy, 72.10% ROC-AUC)
2. **Executive Analytics Dashboard**: High-level KPIs, department-wise attrition distributions, CGPA vs Attendance scatter analysis.
3. **Single Student Diagnostic Engine**: Dynamic risk probability dial, stratified risk tiers (High / Medium / Low), and tailored 3-pillar intervention plans (Academic, Attendance/Habits, Engagement/Financial).
4. **Batch Prediction & Triage Hub**: Upload semester CSV rosters, load 1-click sample batches, filter high-risk cohorts, and dispatch proctor alerts.
5. **Student Roster Explorer**: Search across 19,591 records with department, CGPA, and risk filtering + single student full profile inspector.
6. **Visual Analytics & SHAP Explainability**: Global feature attribution ranking, correlation heatmap, dynamic boxplots and violin distributions.
7. **Interactive What-If Simulator**: Real-time slider adjustments to model the impact of academic recovery, remedial tutoring, and sports participation.
8. **AI Academic Advisor**: Hybrid conversational assistant answering academic counseling queries, risk factor analysis, and VTU viva questions.

---

## 📊 3. Feature Importance & Key Findings (SHAP)
| Rank | Feature | Predictive Weight | Academic Insight |
|---|---|---|---|
| **#1** | `past_failures` | **0.2109** (21.1%) | Students with &ge; 2 backlogs face a 4.2x higher dropout risk. |
| **#2** | `attendance_rate` | **0.1774** (17.7%) | Attendance < 70% is the earliest measurable warning sign. |
| **#3** | `cgpa` | **0.1579** (15.8%) | CGPA < 6.0 indicates 3x higher risk of discontinuation. |
| **#4** | `family_income` | **0.1017** (10.2%) | Socio-economic factor affecting student resource access. |
| **#5** | `study_hours_per_week`| **0.0944** (9.4%) | Direct correlation with assignment submission rate. |
| **#6** | `sports_participation`| **0.0697** (7.0%) | Strong protective shielding against semester fatigue. |

---

## 💻 4. How to Run the Project Locally

### Step 1: Open Terminal in Project Directory
```bash
cd "c:\Users\tarun\OneDrive\Desktop\P project\Student-Dropout-Prediction-using-ML\Natinal_HAK_Final\Natinal_HAK"
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🎯 5. VTU Viva Voce / Presentation Q&A Cheat Sheet

**Q1: What is the objective of this mini project?**  
> *Answer:* To build an AI-powered early warning decision support system that predicts student dropout risk and suggests personalized remedial interventions to enhance academic retention in engineering colleges.

**Q2: Which ML model performs best and why?**  
> *Answer:* Gradient Boosting achieved the highest overall accuracy (73.23%) and balanced precision/ROC-AUC. It excels at capturing complex non-linear combinations (such as low attendance compounded with backlog history).

**Q3: How is data preprocessing handled?**  
> *Answer:* Categorical features (Gender, Department, Scholarship, Parental Education, Extra-curriculars, Sports) are encoded using `LabelEncoder`. Numerical features are scaled using `StandardScaler` fitted on the training split (80-20 stratified split).

**Q4: How does the system help academic proctors / faculty advisors?**  
> *Answer:* It automates batch roster triage, pinpoints at-risk students before semester exams, and generates structured 3-pillar action plans (academic tutorials, attendance notices, and financial aid counseling).
