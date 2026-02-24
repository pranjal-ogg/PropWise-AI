# 🏠 PropWise-AI  
## Intelligent Property Price Prediction System

PropWise-AI is a machine learning–based real estate analytics application that predicts residential property prices using structured housing data. The system uses a complete Scikit-learn pipeline with feature engineering and Random Forest regression, deployed via an interactive Streamlit web interface.

This project was developed as part of an Introduction to Artificial Intelligence & Machine Learning course (Milestone 1).

---

## 🚀 Live Application

🔗 Streamlit App: https://propwise-ai-yr2gt82aogfnwzietyshnq.streamlit.app/ 

---

## 📌 Objective

Estimating property prices accurately is challenging due to multiple influencing factors such as area, rooms, amenities, and furnishing status.

The goal of this project is to:

- Predict property prices using classical machine learning  
- Evaluate model performance  
- Identify key price-driving factors  
- Provide a user-friendly web interface  

---

## 🧠 System Architecture

User Input → Streamlit UI → Preprocessing Pipeline → Random Forest Model → Price Prediction → Result Display

---

## 🛠 Tech Stack

### Machine Learning
- Python
- pandas
- NumPy
- scikit-learn
- joblib

### UI & Visualization
- Streamlit
- matplotlib
- seaborn

### Deployment
- Streamlit Community Cloud

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| MAE | 1,081,837.89 |
| RMSE | 1,474,651.50 |
| R² Score | 0.5698 |

The model demonstrates moderate predictive capability and successfully captures key housing price patterns.

---

## ✨ Key Features

### ✅ Single Property Prediction
Users manually enter property details to receive instant price predictions.

### ✅ Batch Prediction
Upload a CSV file to generate predictions for multiple properties.

### ✅ Model Performance Dashboard
Displays MAE, RMSE, and R² metrics.

### ✅ Feature Importance Visualization
Shows top price-driving factors such as:
- Area
- Total Rooms
- Bathrooms
- Parking

### ✅ Data Explorer
- Dataset preview
- Correlation heatmap
- Price distribution charts

---

## 📂 Project Structure

propwise-ai/

├── app.py  
├── analyze_housing.py  
├── model.pkl  
├── metrics.json  
├── feature_importance.csv  
├── feature_names.joblib  
├── scaler.joblib  
├── housing_model.joblib  
└── requirements.txt  

---

## ⚙️ Installation & Local Setup

### Clone Repository

git clone https://github.com/your-username/propwise-ai.git  
cd propwise-ai  

### Create Virtual Environment

python -m venv venv  
source venv/bin/activate   (Mac/Linux)  
venv\Scripts\activate      (Windows)  

### Install Dependencies

pip install -r requirements.txt  

### Run Application

streamlit run app.py  

---

## 🎯 Milestone 1 Deliverables

- Classical ML pipeline implementation  
- Feature engineering & preprocessing  
- Random Forest regression model  
- Model evaluation (MAE, RMSE, R²)  
- Feature importance analysis  
- Interactive Streamlit deployment  

---

## 🚧 Limitations

- External market trends not included  
- No geospatial modeling  
- R² indicates room for performance improvement  

---

## 🔮 Future Scope (Milestone 2)

Agentic AI Real Estate Advisory System

Planned enhancements:

- Market trend analysis  
- Retrieval-Augmented Generation (RAG)  
- Investment recommendations  
- LLM integration  
- Comparative property analysis  

---

## 👥 Team Members

- Rounak Kumar Saw  
- Pranjal Tripathi
- Priyanshu Verma 

---

## 📜 License

Academic Project – For educational use only.

---

⭐ PropWise-AI  
Smart Decisions. Smarter Investments.
