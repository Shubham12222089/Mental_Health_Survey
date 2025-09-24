# Mental Health Survey ML Pipeline

![ML Pipeline](https://github.com/yourusername/mental-survey-ml/workflows/Mental%20Survey%20ML%20Pipeline/badge.svg)

A simple machine learning pipeline for analyzing mental health survey data to predict treatment needs. Uses scikit-learn, MLflow for experiment tracking, and GitHub Actions for CI/CD.

## 🎯 Project Overview

This project analyzes mental health survey data to predict whether someone might need mental health treatment based on workplace and personal factors using Random Forest classification.

## 🏗️ Project Structure

```
mental-survey-ml/
├── .github/workflows/
│   └── ml-pipeline.yml           # GitHub Actions workflow
├── data/
│   ├── raw/survey_data.csv       # Raw survey data
│   └── processed/                # Processed data files
├── src/
│   ├── preprocess.py             # Data preprocessing
│   ├── train.py                  # Model training with MLflow
│   └── evaluate.py               # Model evaluation
├── Models/model.pkl              # Trained model
├── metrics/                      # Evaluation results
├── mlruns/                       # MLflow tracking
├── params.yaml                   # Simple configuration
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

1. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/mental-survey-ml.git
   cd mental-survey-ml
   pip install -r requirements.txt
   ```

2. **Create sample data** (or add your own to `data/raw/survey_data.csv`)
   ```bash
   mkdir -p data/raw
   python -c "
   import pandas as pd
   import numpy as np
   np.random.seed(42)
   data = {
       'Age': np.random.randint(18, 80, 1000),
       'Gender': np.random.choice(['Male', 'Female'], 1000),
       'family_history': np.random.choice([0, 1], 1000),
       'work_interfere': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often'], 1000),
       'treatment': np.random.choice([0, 1], 1000)
   }
   df = pd.DataFrame(data)
   df.to_csv('data/raw/survey_data.csv', index=False)
   "
   ```

3. **Run the pipeline**
   ```bash
   python src/preprocess.py  # Process data
   python src/train.py       # Train model
   python src/evaluate.py    # Evaluate model
   ```

## 📊 What It Does

- **Preprocessing**: Cleans data, handles missing values, encodes categories
- **Training**: Trains Random Forest model with MLflow tracking
- **Evaluation**: Tests model performance and creates visualizations
- **CI/CD**: Automates everything with GitHub Actions

## 🔧 Configuration

Edit `params.yaml` to change model settings:

```yaml
# Model parameters
n_estimators: 100
max_depth: 10
random_state: 42

# Data parameters
test_size: 0.2
```

## 📈 MLflow Tracking

View experiment results:
```bash
mlflow ui
```
Then open http://localhost:5000 to see your experiments.

## 🤖 GitHub Actions

The workflow runs automatically on pushes to main:
- Creates sample data
- Runs preprocessing
- Trains model with MLflow
- Evaluates performance
- Saves results as artifacts

## 📋 Expected Data Format

Your CSV should have:
- Feature columns (Age, Gender, family_history, etc.)
- Target column named `treatment` (0 or 1)

## 🚀 Results

The pipeline creates:
- `Models/model.pkl` - Trained model
- `metrics/confusion_matrix.png` - Performance visualization
- MLflow experiment logs with metrics and parameters

## 🤝 Contributing

1. Fork the repo
2. Make changes
3. Test locally: `python src/preprocess.py && python src/train.py && python src/evaluate.py`
4. Submit pull request

---

Simple, clean, and effective ML pipeline for mental health prediction! 🧠✨