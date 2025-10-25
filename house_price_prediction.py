"""
House Price Prediction Project
==============================
Goal: Predict house prices based on features like area, rooms, location, etc.
Tech Stack: Python, Pandas, NumPy, Matplotlib, Scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HOUSE PRICE PREDICTION PROJECT")
print("=" * 60)

# Step 1: Generate Synthetic Dataset (since we don't have access to external datasets)
print("\n1. GENERATING SYNTHETIC DATASET")
print("-" * 40)

np.random.seed(42)
n_samples = 1000

# Generate synthetic house data
data = {
    'area': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'garage': np.random.randint(0, 3, n_samples),
    'location': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples),
    'condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate realistic house prices based on features
base_price = 100000
price = (
    base_price +
    df['area'] * 150 +
    df['bedrooms'] * 20000 +
    df['bathrooms'] * 15000 +
    (50 - df['age']) * 2000 +
    df['garage'] * 10000 +
    df['location'].map({'Downtown': 50000, 'Suburb': 20000, 'Rural': 0}) +
    df['condition'].map({'Excellent': 30000, 'Good': 15000, 'Fair': 5000, 'Poor': -10000}) +
    np.random.normal(0, 20000, n_samples)  # Add some noise
)

df['price'] = np.maximum(price, 50000)  # Ensure minimum price

print(f"Dataset created with {len(df)} samples and {len(df.columns)} features")
print("Features:", list(df.columns))

# Step 2: Data Exploration
print("\n2. DATA EXPLORATION")
print("-" * 40)

print("\nDataset Head:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Data Preprocessing
print("\n3. DATA PREPROCESSING")
print("-" * 40)

# Handle categorical variables
print("Encoding categorical variables...")
le_location = LabelEncoder()
le_condition = LabelEncoder()

df['location_encoded'] = le_location.fit_transform(df['location'])
df['condition_encoded'] = le_condition.fit_transform(df['condition'])

# Create feature matrix
features = ['area', 'bedrooms', 'bathrooms', 'age', 'garage', 'location_encoded', 'condition_encoded']
X = df[features]
y = df['price']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Step 4: Exploratory Data Analysis (EDA)
print("\n4. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('House Price Analysis - EDA', fontsize=16, fontweight='bold')

# Price distribution
axes[0, 0].hist(df['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Price Distribution')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')

# Area vs Price scatter plot
axes[0, 1].scatter(df['area'], df['price'], alpha=0.6, color='coral')
axes[0, 1].set_title('Area vs Price')
axes[0, 1].set_xlabel('Area (sq ft)')
axes[0, 1].set_ylabel('Price ($)')

# Price by Location
location_prices = df.groupby('location')['price'].mean()
axes[1, 0].bar(location_prices.index, location_prices.values, color=['gold', 'lightgreen', 'lightcoral'])
axes[1, 0].set_title('Average Price by Location')
axes[1, 0].set_xlabel('Location')
axes[1, 0].set_ylabel('Average Price ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Bedrooms vs Price
bedroom_prices = df.groupby('bedrooms')['price'].mean()
axes[1, 1].bar(bedroom_prices.index, bedroom_prices.values, color='lightblue')
axes[1, 1].set_title('Average Price by Bedrooms')
axes[1, 1].set_xlabel('Number of Bedrooms')
axes[1, 1].set_ylabel('Average Price ($)')

plt.tight_layout()
plt.savefig('c:\\Users\\shiva\\OneDrive\\Documents\\Desktop\\ML\\House price prediction\\eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(10, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('c:\\Users\\shiva\\OneDrive\\Documents\\Desktop\\ML\\House price prediction\\correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 5: Data Splitting
print("\n5. DATA SPLITTING")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 6: Feature Scaling
print("\n6. FEATURE SCALING")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler")

# Step 7: Model Training
print("\n7. MODEL TRAINING")
print("-" * 40)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Linear Regression model trained successfully!")

# Step 8: Model Evaluation
print("\n8. MODEL EVALUATION")
print("-" * 40)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("TRAINING METRICS:")
print(f"R² Score: {train_r2:.4f}")
print(f"MAE: ${train_mae:,.2f}")
print(f"RMSE: ${train_rmse:,.2f}")

print("\nTESTING METRICS:")
print(f"R² Score: {test_r2:.4f}")
print(f"MAE: ${test_mae:,.2f}")
print(f"RMSE: ${test_rmse:,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nFEATURE IMPORTANCE (by coefficient magnitude):")
print(feature_importance)

# Step 9: Visualization of Results
print("\n9. RESULTS VISUALIZATION")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Model Performance Visualization', fontsize=16, fontweight='bold')

# Actual vs Predicted (Training)
axes[0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].set_title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Actual vs Predicted (Testing)
axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='green', label='Testing')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Price ($)')
axes[1].set_ylabel('Predicted Price ($)')
axes[1].set_title(f'Testing Set: Actual vs Predicted\nR² = {test_r2:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('c:\\Users\\shiva\\OneDrive\\Documents\\Desktop\\ML\\House price prediction\\model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 10: Sample Predictions
print("\n10. SAMPLE PREDICTIONS")
print("-" * 40)

sample_indices = np.random.choice(X_test.index, 5, replace=False)
sample_data = X_test.loc[sample_indices]
sample_actual = y_test.loc[sample_indices]
sample_predictions = model.predict(scaler.transform(sample_data))

print("Sample Predictions vs Actual Prices:")
print("-" * 50)
for i, idx in enumerate(sample_indices):
    actual = sample_actual.iloc[i]
    predicted = sample_predictions[i]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100
    
    print(f"Sample {i+1}:")
    print(f"  Features: Area={sample_data.loc[idx, 'area']:.0f}, Bedrooms={sample_data.loc[idx, 'bedrooms']}, "
          f"Bathrooms={sample_data.loc[idx, 'bathrooms']}, Age={sample_data.loc[idx, 'age']}")
    print(f"  Actual Price: ${actual:,.2f}")
    print(f"  Predicted Price: ${predicted:,.2f}")
    print(f"  Error: ${error:,.2f} ({error_pct:.1f}%)")
    print()

# Step 11: Save the Model
print("\n11. SAVING THE MODEL")
print("-" * 40)

# Save the trained model and scaler
model_data = {
    'model': model,
    'scaler': scaler,
    'feature_names': features,
    'label_encoders': {
        'location': le_location,
        'condition': le_condition
    },
    'metrics': {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }
}

joblib.dump(model_data, 'c:\\Users\\shiva\\OneDrive\\Documents\\Desktop\\ML\\House price prediction\\house_price_model.pkl')
print("Model saved successfully as 'house_price_model.pkl'")

# Save feature importance
feature_importance.to_csv('c:\\Users\\shiva\\OneDrive\\Documents\\Desktop\\ML\\House price prediction\\feature_importance.csv', index=False)
print("Feature importance saved as 'feature_importance.csv'")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"Final Model Performance:")
print(f"- R² Score: {test_r2:.4f}")
print(f"- Mean Absolute Error: ${test_mae:,.2f}")
print(f"- Root Mean Square Error: ${test_rmse:,.2f}")
print("\nFiles Generated:")
print("- house_price_model.pkl (trained model)")
print("- feature_importance.csv (feature analysis)")
print("- eda_analysis.png (exploratory data analysis)")
print("- correlation_heatmap.png (feature correlations)")
print("- model_performance.png (prediction accuracy)")
