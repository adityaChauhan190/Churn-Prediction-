# 📉 Customer Churn Prediction Dashboard  

A professional **machine learning pipeline** and **Streamlit web app** that predicts customer churn using advanced data preprocessing, hyperparameter tuning, and an optimized **Random Forest Classifier**.  

This project not only provides predictions but also **explains feature importance** through visualizations, offering actionable insights for business stakeholders.  

---

## 🚀 Features  

- ✅ End-to-end ML pipeline with preprocessing & Random Forest Classifier  
- ✅ Hyperparameter tuning using **GridSearchCV** for best model performance  
- ✅ Achieved **ROC-AUC = 0.867** on validation data  
- ✅ Interactive **Streamlit dashboard** with a modern UI (professional look, not default)  
- ✅ Glassmorphism-inspired **prediction card** for churn results  
- ✅ **Feature importance visualization** to explain key drivers of churn  
- ✅ Supports both **light/dark themes**  

---

## 🏗️ Tech Stack  

- **Python 3.9+**  
- **Scikit-learn** – Model training & pipeline  
- **Pandas / NumPy** – Data preprocessing  
- **Matplotlib** – Visualization  
- **Streamlit** – Frontend dashboard  
- **Joblib** – Model persistence  

---

## 📂 Project Structure  
├── data/ # Dataset (not included in repo for privacy)
├── models/
│ └── churn_model.pkl # Trained pipeline (Random Forest Classifier)
├── app.py # Streamlit application
├── train_model.ipynb # Jupyter notebook for training & experiments
├── requirements.txt # Project dependencies
└── README.md # Project documentation

## 📊 Model Training

**The model pipeline:**

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

rf_pipe = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('rf_clf', RandomForestClassifier(
            n_estimators=500,    
            max_depth=10,       
            class_weight='balanced', 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]
)
rf_pipe.fit(X_train, y_train)


Final ROC-AUC Score: 0.867
Balanced for class imbalance
Optimized for interpretability & accuracy

## 🖥️ Running the Streamlit App

**Start the dashboard:**

streamlit run app.py


Your app will be available at:
👉 http://localhost:8501

## 🎨 Dashboard Preview

Left panel: Customer input details
Right panel: Prediction results & feature importance chart
Prediction Card: Glassmorphism UI with churn probability
