from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import json
from io import BytesIO
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Load model artifacts
print("Loading model artifacts...")
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
print("Model loaded successfully!")

# Load dataset
df = pd.read_csv('cleaned_dataset.csv')

# Feature columns
FEATURE_COLUMNS = ['gender', 'age', 'department', 'cgpa', 'attendance_rate',
                   'study_hours_per_week', 'assignments_submitted', 'past_failures',
                   'extra_curricular', 'sports_participation', 'projects_completed',
                   'family_income', 'parental_education', 'scholarship', 'total_activities']

def get_risk_level(probability):
    """Determine risk level based on dropout probability"""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        "LOW": "#00FF00",
        "MEDIUM": "#FFA500",
        "HIGH": "#FF0000"
    }
    return colors.get(risk_level, "#000000")

@app.route('/')
def index():
    """Home/Dashboard page"""
    # Calculate statistics
    total_students = len(df)
    dropout_count = df['dropout'].sum()
    dropout_rate = (dropout_count / total_students) * 100

    # Department-wise stats
    dept_stats = df.groupby('department')['dropout'].agg(['count', 'sum']).reset_index()
    dept_stats['dropout_rate'] = (dept_stats['sum'] / dept_stats['count']) * 100
    dept_stats = dept_stats.sort_values('dropout_rate', ascending=False)

    stats = {
        'total_students': total_students,
        'dropout_count': int(dropout_count),
        'dropout_rate': round(dropout_rate, 2),
        'retention_rate': round(100 - dropout_rate, 2),
        'avg_cgpa': round(df['cgpa'].mean(), 2),
        'avg_attendance': round(df['attendance_rate'].mean(), 2),
        'dept_stats': dept_stats.to_dict('records')
    }

    return render_template('index.html', stats=stats)

@app.route('/predict')
def predict_page():
    """Single prediction page"""
    return render_template('predict.html')

@app.route('/batch')
def batch_page():
    """Batch prediction page"""
    return render_template('batch.html')

@app.route('/students')
def students_page():
    """View all students"""
    students = df.to_dict('records')
    return render_template('students.html', students=students)

@app.route('/analytics')
def analytics_page():
    """Analytics dashboard"""
    return render_template('analytics.html')

@app.route('/simulator')
def simulator_page():
    """What-if simulator"""
    return render_template('simulator.html')

@app.route('/chatbot')
def chatbot_page():
    """AI Chatbot"""
    return render_template('chatbot.html')

# API Endpoints
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Single prediction API"""
    try:
        data = request.json

        # Create dataframe
        input_data = pd.DataFrame([data])

        # Encode categorical variables
        for col in label_encoders.keys():
            if col in input_data.columns:
                le = label_encoders[col]
                input_data[col] = le.transform(input_data[col])

        # Ensure correct column order
        input_data = input_data[FEATURE_COLUMNS]

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])
        risk_level = get_risk_level(probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'dropout_probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'risk_color': get_risk_color(risk_level)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """Batch prediction API"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']

        # Read CSV
        batch_df = pd.read_csv(file)

        # Prepare features
        X_batch = batch_df[FEATURE_COLUMNS].copy()

        # Encode
        for col in label_encoders.keys():
            if col in X_batch.columns:
                le = label_encoders[col]
                X_batch[col] = le.transform(X_batch[col])

        # Scale
        X_batch_scaled = scaler.transform(X_batch)

        # Predict
        predictions = model.predict(X_batch_scaled)
        probabilities = model.predict_proba(X_batch_scaled)[:, 1]

        # Add results to dataframe
        batch_df['dropout_prediction'] = predictions
        batch_df['dropout_probability'] = probabilities * 100
        batch_df['risk_level'] = [get_risk_level(p) for p in probabilities]

        # Statistics
        high_risk_count = sum(batch_df['risk_level'] == 'HIGH')
        medium_risk_count = sum(batch_df['risk_level'] == 'MEDIUM')
        low_risk_count = sum(batch_df['risk_level'] == 'LOW')

        return jsonify({
            'success': True,
            'total_records': len(batch_df),
            'high_risk': high_risk_count,
            'medium_risk': medium_risk_count,
            'low_risk': low_risk_count,
            'results': batch_df.to_dict('records')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/statistics')
def api_statistics():
    """Get overall statistics"""
    stats = {
        'total_students': len(df),
        'dropouts': int(df['dropout'].sum()),
        'dropout_rate': round((df['dropout'].sum() / len(df)) * 100, 2),
        'avg_cgpa': round(df['cgpa'].mean(), 2),
        'avg_attendance': round(df['attendance_rate'].mean(), 2),
        'avg_study_hours': round(df['study_hours_per_week'].mean(), 2)
    }
    return jsonify(stats)

@app.route('/api/department-stats')
def api_department_stats():
    """Get department-wise statistics"""
    dept_stats = df.groupby('department').agg({
        'dropout': ['count', 'sum', 'mean'],
        'cgpa': 'mean',
        'attendance_rate': 'mean'
    }).reset_index()

    dept_stats.columns = ['department', 'total', 'dropouts', 'dropout_rate', 'avg_cgpa', 'avg_attendance']
    dept_stats['dropout_rate'] = dept_stats['dropout_rate'] * 100

    return jsonify(dept_stats.to_dict('records'))

@app.route('/api/feature-importance')
def api_feature_importance():
    """Get feature importance"""
    try:
        shap_fi = pd.read_csv('models/shap_feature_importance.csv')
        return jsonify(shap_fi.to_dict('records'))
    except:
        return jsonify({'error': 'Feature importance data not available'}), 404

@app.route('/api/chatbot', methods=['POST'])
def api_chatbot():
    """Chatbot API"""
    data = request.json
    user_message = data.get('message', '').lower()

    # Simple rule-based responses
    if 'dropout' in user_message or 'risk' in user_message:
        response = "Student dropout is influenced by multiple factors including academic performance (CGPA), attendance rate, past failures, and engagement in extracurricular activities. Our model predicts dropout risk based on these factors."
    elif 'cgpa' in user_message or 'grade' in user_message:
        response = "CGPA is one of the most important factors. Students with CGPA below 6.0 have 3x higher dropout risk. Maintaining a CGPA above 7.0 significantly reduces dropout probability."
    elif 'attendance' in user_message:
        response = "Attendance rate is critical! Students with attendance below 70% face significantly higher dropout risk. Regular attendance (>80%) is strongly associated with successful completion."
    elif 'scholarship' in user_message:
        response = "Scholarship holders show 30% lower dropout rates. Financial support helps students focus on academics and reduces dropout risk significantly."
    elif 'activities' in user_message or 'sports' in user_message:
        response = "Extra-curricular activities and sports participation reduce dropout risk by 40%! Engaged students are more connected to campus and more likely to persist."
    elif 'help' in user_message or 'improve' in user_message:
        response = "To reduce dropout risk: 1) Maintain CGPA above 7.0, 2) Keep attendance above 80%, 3) Participate in extracurricular activities, 4) Seek help early for academic difficulties, 5) Apply for scholarships if eligible."
    else:
        response = "I can help you understand dropout risk factors. Ask me about CGPA, attendance, scholarships, activities, or how to improve student retention!"

    return jsonify({
        'success': True,
        'response': response
    })

if __name__ == '__main__':
    print("=" * 80)
    print("FLASK APPLICATION STARTING")
    print("=" * 80)
    print("\nOpen your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5000)
