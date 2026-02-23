import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(layout="wide", page_title="PropWise AI - Property Price Predictor")

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Artifact Loading
def load_artifacts():
    try:
        pipeline = joblib.load('model.pkl')
    except Exception as e:
        st.warning(f"Model mismatch detected ({e}). Rebuilding the model locally...")
        # Automatically retrain if the pickle is incompatible
        os.system("python analyze_housing.py")
        try:
            pipeline = joblib.load('model.pkl')
        except Exception as e2:
            st.error(f"Failed to rebuild and load model: {e2}")
            return None, {}, None

    try:
        metrics = {}
        if os.path.exists('metrics.json'):
            with open('metrics.json', 'r') as f:
                metrics = json.load(f)
        df_imp = pd.read_csv('feature_importance.csv') if os.path.exists('feature_importance.csv') else None
        return pipeline, metrics, df_imp
    except Exception as e:
        st.error(f"Error loading secondary artifacts: {e}")
        return pipeline, {}, None

# Feature Engineering
def engineer_features(df):
    current_year = datetime.now().year
    if 'year_built' not in df.columns:
        df['year_built'] = current_year 
    df['property_age'] = current_year - df['year_built']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    return df

# --- PAGE: Home ---
def render_home():
    st.title("🏠 PropWise AI: Property Price Predictor")
    st.markdown("---")
    st.markdown("""
    ### Welcome to PropWise AI
    PropWise AI is an advanced machine learning dashboard designed to revolutionize how property evaluations are conducted. By combining spatial attributes, structural features, and modern amenities, our system delivers high-precision market valuations in seconds.
    
    #### 📂 Dataset Highlights
    The engine is trained on the comprehensive Housing dataset, focusing on:
    - **Spatial Characteristics**: Total area (sq ft), number of stories, and parking capacity.
    - **Structural Details**: Room distribution (bedrooms and bathrooms).
    - **Premium Features**: Air conditioning, preferred area status, and accessibility.
    - **Refurbishment Status**: Current furnishing condition.
    
    #### ⚙️ The Pipeline Workflow
    1. **Data Ingestion**: Loading raw historical housing data.
    2. **Transformation**: Automated feature engineering.
    3. **Preprocessing**: One-Hot Encoding and Scaling.
    4. **Modeling**: Random Forest Regressor Pipeline.
    
    #### 🚀 How to use
    1. **Explore Data**: Navigate to **Data Explorer** to view the dataset structure and correlation heatmaps.
    2. **Get Valuations**: Go to **Predict Price** to enter property details for an instant valuation.
    3. **Batch Processing**: Upload a CSV file via the **Sidebar** or the Predict page to process multiple properties at once.
    4. **Analyze Performance**: Check **Model Performance** to understand the accuracy and key factors driving property prices.
    """)

# --- PAGE: Data Explorer ---
def render_data_explorer():
    st.title("🔍 Data Explorer")
    st.markdown("---")
    
    data_path = os.path.join("data", "Housing.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📋 Dataset Overview")
            st.write(f"**Total Properties:** {df.shape[0]}")
            st.write(f"**Total Features:** {df.shape[1]}")
            st.dataframe(df.head(5), use_container_width=True)
            
        with col2:
            st.subheader("📈 Quick Statistics")
            st.dataframe(df.describe().T, use_container_width=True)
            
        st.markdown("---")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("🔥 Feature Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            plt.tight_layout()
            st.pyplot(fig_corr)
            
        with col_viz2:
            st.subheader("💰 Price Distribution Analysis")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 8))
            sns.histplot(df['price'], kde=True, color='#007bff', ax=ax_dist)
            ax_dist.set_title("Property Price Distribution")
            ax_dist.set_xlabel("Price")
            plt.tight_layout()
            st.pyplot(fig_dist)
    else:
        st.warning("Housing dataset not found. Please ensure `data/Housing.csv` is correctly placed.")

# --- PAGE: Predict Price ---
def render_predict_price(pipeline, sidebar_file=None):
    st.title("📍 Predict Property Price")
    st.markdown("---")
    
    if pipeline is None:
        st.error("Model artifacts not found. Please ensure training has been completed.")
        return

    tab1, tab2 = st.tabs(["Single Property valuation", "Batch Batch Prediction (Optional)"])
    
    with tab1:
        st.subheader("Property Specification Form")
        with st.form("valuation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                area = st.number_input("Total Area (sq ft)", value=5000, step=100)
                bedrooms = st.number_input("Number of Bedrooms", value=3, min_value=1, max_value=10)
                bathrooms = st.number_input("Number of Bathrooms", value=2, min_value=1, max_value=5)
                stories = st.number_input("Total Stories", value=2, min_value=1, max_value=4)
                parking = st.number_input("Parking Capacity", value=1, min_value=0, max_value=3)
                furnishingstatus = st.selectbox("Current Furnishing", ["furnished", "semi-furnished", "unfurnished"])
                
            with col2:
                mainroad = st.selectbox("Main Road Access", ["yes", "no"])
                guestroom = st.selectbox("Guestroom Availability", ["yes", "no"])
                basement = st.selectbox("Basement Level", ["yes", "no"])
                airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
                prefarea = st.selectbox("Preferred Location", ["yes", "no"])
                hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
            
            st.markdown(" ")
            submit = st.form_submit_button("💰 Get Instant Valuation")
            
        if submit:
            input_dict = {
                'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 
                'stories': stories, 'mainroad': mainroad, 'guestroom': guestroom,
                'basement': basement, 'airconditioning': airconditioning, 
                'parking': parking, 'prefarea': prefarea, 'furnishingstatus': furnishingstatus,
                'hotwaterheating': hotwaterheating, 'year_built': datetime.now().year
            }
            input_df = engineer_features(pd.DataFrame([input_dict]))
            prediction = pipeline.predict(input_df)[0]
            
            st.success(f"### Estimated Market Valuation: ${prediction:,.2f}")
            st.markdown("---")
            st.info("This prediction is generated based on current model training on historical trends.")

    with tab2:
        st.subheader("📋 Batch Processing Pipeline")
        
        # Determine which file to use
        current_file = sidebar_file if sidebar_file is not None else st.file_uploader("Upload CSV property list", type=["csv"], key="batch_uploader")
        
        if current_file:
            df_input = pd.read_csv(current_file)
            st.info(f"Loaded {len(df_input)} properties for analysis.")
            st.dataframe(df_input.head(10), use_container_width=True)
            
            if st.button("🚀 Process Batch valuation"):
                X_batch = engineer_features(df_input.copy())
                results = df_input.copy()
                results['Predicted Price'] = pipeline.predict(X_batch)
                results['Formatted Price'] = results['Predicted Price'].apply(lambda x: f"${x:,.2f}")
                
                st.subheader("✅ Processed Results")
                st.dataframe(results, use_container_width=True)
                
                csv_data = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download All Predictions", 
                    csv_data, 
                    "property_valuations.csv", 
                    "text/csv"
                )
        else:
            st.info("Please upload a CSV file via the sidebar or the box above to begin batch processing.")

# --- PAGE: Model Performance ---
def render_model_performance(metrics, df_imp):
    st.title("📈 Model Performance Monitoring")
    st.markdown("---")
    
    if metrics:
        st.subheader("🎯 Key Performance Indicators")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:,.0f}")
        with m_col2:
            st.metric("Root Mean Squared Error (RMSE)", f"${metrics['RMSE']:,.0f}")
        with m_col3:
            st.metric("R2 Variance Score", f"{metrics['R2']:.4f}")
    else:
        st.warning("Performance metrics file `metrics.json` missing.")
        
    st.markdown("---")
    
    if df_imp is not None:
        st.subheader("🔍 Top Price-Driving Factors")
        st.markdown("These features have the highest relative impact on the property valuation.")
        
        top_10 = df_imp.head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, 10))
        
        ax.barh(top_10['Feature'][::-1], top_10['Importance'][::-1], color=colors)
        ax.set_xlabel('Relative Importance')
        ax.set_title('Top 10 Feature Importance (Random Forest)')
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.expander("Explore Full Feature Statistics"):
            st.dataframe(df_imp, use_container_width=True)
    else:
        st.warning("Feature importance data missing.")

# --- MAIN EXECUTION ---
def main():
    # Load persistence
    pipeline, metrics, df_imp = load_artifacts()
    
    # Sidebar Navigation
    st.sidebar.title("PropWise AI Dashboard")
    st.sidebar.markdown("*Empowering property intelligence*")
    st.sidebar.markdown("---")
    
    navigation = st.sidebar.radio(
        "Navigation Menu", 
        ["Home", "Data Explorer", "Predict Price", "Model Performance"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Data for Prediction")
    sidebar_file = st.sidebar.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
    
    st.sidebar.markdown("---")
    
    # Dispatcher
    if navigation == "Home":
        render_home()
    elif navigation == "Data Explorer":
        render_data_explorer()
    elif navigation == "Predict Price":
        render_predict_price(pipeline, sidebar_file)
    elif navigation == "Model Performance":
        render_model_performance(metrics, df_imp)

if __name__ == "__main__":
    main()
