import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Read dataset
df = pd.read_excel('dataset.xlsx')

print("=" * 80)
print("INITIAL DATA OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData Info:")
print(df.info())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

print("\n" + "=" * 80)
print("DATA CLEANING")
print("=" * 80)

# Standardize gender values
df['gender'] = df['gender'].str.upper().replace({'M': 'MALE', 'F': 'FEMALE'})
print(f"Gender unique values: {df['gender'].unique()}")

# Standardize sports_participation
df['sports_participation'] = df['sports_participation'].str.upper().replace({'Y': 'YES', 'N': 'NO'})
print(f"Sports participation unique values: {df['sports_participation'].unique()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Check target distribution
print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)
print(df['dropout'].value_counts())
print(f"\nDropout percentage: {df['dropout'].mean() * 100:.2f}%")

# Save cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
print("\n✓ Cleaned dataset saved to 'cleaned_dataset.csv'")

# Basic statistics
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
print(df.describe())

# Correlation analysis for numerical features
print("\n" + "=" * 80)
print("CORRELATION WITH DROPOUT")
print("=" * 80)
numerical_cols = ['age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures',
                  'study_hours_per_week', 'assignments_submitted', 'projects_completed',
                  'total_activities']
correlations = df[numerical_cols + ['dropout']].corr()['dropout'].sort_values(ascending=False)
print(correlations)

print("\n" + "=" * 80)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 80)
categorical_cols = ['gender', 'department', 'scholarship', 'parental_education',
                    'extra_curricular', 'sports_participation']
for col in categorical_cols:
    print(f"\n{col}:")
    print(df.groupby(col)['dropout'].agg(['count', 'mean']).sort_values('mean', ascending=False))

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETE")
print("=" * 80)
