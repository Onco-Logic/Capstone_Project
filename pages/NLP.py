import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ─────────────────────────────────────────────
# Step 1: Load and Inspect the Data
# ─────────────────────────────────────────────
df = pd.read_csv("Data/TCGA_Reports.csv")

# Basic info
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Preview some reports
print("\nSample reports:")
print(df["report_text"].head())

# Report length stats
df["report_length"] = df["report_text"].fillna("").apply(len)
print("\nReport length statistics:")
print(df["report_length"].describe())

# Class distribution
if "cancer_type" in df.columns:
    print("\nCancer type distribution:")
    print(df["cancer_type"].value_counts())

# ─────────────────────────────────────────────
# Step 2: Clean Text
# ─────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned"] = df["report_text"].fillna("").apply(clean_text)

# ─────────────────────────────────────────────
# Step 3: TF-IDF Vectorization
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(df["cleaned"])
print("\nTF-IDF matrix shape:", X_tfidf.shape)