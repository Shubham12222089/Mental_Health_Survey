import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def preprocess():
    """Simple data preprocessing function"""
    
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Load raw data
    print("Loading raw survey data...")
    df = pd.read_csv("data/raw/survey_data.csv")
    print(f"Loaded {len(df)} records")
    
    # Basic data cleaning
    # Remove rows with missing target
    if 'treatment' in df.columns:
        df = df.dropna(subset=['treatment'])
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'treatment' in categorical_cols:
        categorical_cols.remove('treatment')
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Ensure treatment is numeric
    if 'treatment' in df.columns and df['treatment'].dtype == 'object':
        le = LabelEncoder()
        df['treatment'] = le.fit_transform(df['treatment'])
    
    # Save processed data
    print("Saving processed data...")
    df.to_csv("data/processed/survey_data.csv", index=False)
    
    # Split into train/test
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    
    print(f"Preprocessing complete!")
    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")
    
    return df

if __name__ == "__main__":
    preprocess()
