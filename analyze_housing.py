import pandas as pd
import os
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_and_engineer_data(file_path):
    """
    Loads data and performs feature engineering.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    df = pd.read_csv(file_path)
    
    # Random seed for synthetic year_built
    np.random.seed(42)
    df['year_built'] = np.random.randint(1950, 2024, size=len(df))
    
    # Feature Engineering
    current_year = datetime.now().year
    df['property_age'] = current_year - df['year_built']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Note: price_per_sqft is excluded to avoid data leakage
    return df

def train_pipeline(df):
    """
    Builds, trains and saves a Scikit-learn Pipeline.
    """
    print("--- Starting Pipeline Training ---")
    
    # Define features and target
    X = df.drop(columns=['price'])
    # Explicitly remove price_per_sqft if it exists (leakage)
    if 'price_per_sqft' in X.columns:
        X = X.drop(columns=['price_per_sqft'])
        
    y = df['price']

    # Identify columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    print(f"Categorical Features: {categorical_cols}")
    print(f"Numerical Features: {numerical_cols}")

    # Create Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Create Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("Fitting the pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(mae, 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }

    print("\n--- Evaluation Metrics ---")
    print(json.dumps(metrics, indent=4))

    # Save artifacts
    print("\nSaving model.pkl...")
    joblib.dump(pipeline, 'model.pkl')
    
    print("Saving metrics.json...")
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("--- Training Complete ---\n")
    return pipeline, X.columns.tolist()

def plot_importance(pipeline):
    """
    Extracts, prints top 10, and saves feature importance to CSV and plot.
    """
    print("--- Extracting and Saving Feature Importance ---")
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get all feature names after transformation
    all_features = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    
    # Create DataFrame and sort
    df_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Print top 10
    print("\nTop 10 Most Important Features:")
    print(df_importance.head(10).to_string(index=False))
    
    # Save to CSV
    df_importance.to_csv('feature_importance.csv', index=False)
    print("\nFeature importance saved to: feature_importance.csv")

    # Plot top 10
    num_show = min(10, len(importances))
    top_features = df_importance.head(num_show)
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances (Pipeline Model)")
    plt.barh(range(num_show), top_features['Importance'][::-1], color='skyblue', align='center')
    plt.yticks(range(num_show), top_features['Feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Feature importance plot updated.\n")

if __name__ == "__main__":
    data_path = os.path.join("data", "Housing.csv")
    df = load_and_engineer_data(data_path)
    
    if df is not None:
        pipeline, feature_names = train_pipeline(df)
        plot_importance(pipeline)
