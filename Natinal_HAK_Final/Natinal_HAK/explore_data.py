import pandas as pd
import PyPDF2

# Read Excel dataset
df = pd.read_excel('dataset.xlsx')
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nBasic statistics:")
print(df.describe())

# Read PDF
print("\n" + "=" * 80)
print("PDF CONTENT")
print("=" * 80)
with open('Tech Symposium Hackathon.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    print(f"\nTotal pages: {len(pdf_reader.pages)}\n")
    for i, page in enumerate(pdf_reader.pages):
        print(f"--- Page {i+1} ---")
        print(page.extract_text())
        print()
