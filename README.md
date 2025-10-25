# House-price-Predictor

## Project Overview
This project demonstrates end-to-end machine learning workflow from data generation to deployment. It predicts house prices based on features like area, bedrooms, location, and condition using a trained Linear Regression model.

## Key Features

- ** Machine Learning Pipeline**: Complete ML workflow with data preprocessing, training, and evaluation
- ** Interactive Web App**: Beautiful Streamlit interface for real-time predictions
- ** Static Web Version**: Netlify-deployable HTML/CSS/JS version
- ** Data Visualization**: Comprehensive EDA with matplotlib and seaborn
- ** Model Persistence**: Trained model saved with joblib for reuse
- ** Responsive Design**: Works seamlessly on desktop and mobile

## Live Demo

🔗 **[Try the Live App](https://house-price-predicter.netlify.app/)**

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Machine learning algorithms |
| **Matplotlib/Seaborn** | Data visualization |
| **Streamlit** | Interactive web application |
| **HTML/CSS/JS** | Static web deployment |
| **Netlify** | Web hosting and deployment |

## Dataset Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| **Area** | Numerical | Living area in square feet | 500-5000 |
| **Bedrooms** | Numerical | Number of bedrooms | 1-5 |
| **Bathrooms** | Numerical | Number of bathrooms | 1-4 |
| **Age** | Numerical | Age of house in years | 0-50 |
| **Garage** | Numerical | Number of garage spaces | 0-3 |
| **Location** | Categorical | Location type | Downtown, Suburb, Rural |
| **Condition** | Categorical | House condition | Excellent, Good, Fair, Poor |

## Quick Start

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the ML pipeline**
   ```bash
   python house_price_prediction.py
   ```

4. **Launch the web app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   ```
   http://localhost:8501
   ```
   
## Machine Learning Pipeline

### 1. Data Generation
- Creates 1000 synthetic house records
- Realistic price calculation based on features
- Controlled noise for model training

### 2. Data Preprocessing
```python
# Categorical encoding
location_encoded = LabelEncoder().fit_transform(location)
condition_encoded = LabelEncoder().fit_transform(condition)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Model Training
```python
# Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

### 4. Model Evaluation
- **R² Score**: ~0.91 (91% variance explained)
- **MAE**: ~$22,800 (Mean Absolute Error)
- **RMSE**: ~$28,100 (Root Mean Square Error)

## Model Performance

| Metric | Training | Testing |
|--------|----------|---------|
| **R² Score** | 0.9164 | 0.9103 |
| **MAE** | $21,254 | $22,803 |
| **RMSE** | $26,227 | $28,114 |

### Feature Importance

| Feature | Coefficient | Impact |
|---------|-------------|--------|
| **Area** | 72,395 | Highest impact |
| **Age** | -30,132 | Negative (older = cheaper) |
| **Bedrooms** | 29,241 | Strong positive |
| **Bathrooms** | 13,402 | Moderate positive |
| **Condition** | -12,336 | Quality matters |
| **Location** | -12,299 | Location premium |
| **Garage** | 7,279 | Additional value |

## Deployment Options

### Option 1: Netlify (Static)
```bash
python build_static.py
# Drag dist/ folder to netlify.com
```

### Option 2: Streamlit Cloud
```bash
# Connect GitHub repo to Streamlit Cloud
# Auto-deploys on push
```

### Option 3: Heroku
```bash
# Add Procfile and deploy to Heroku
```

## Usage Examples

### Programmatic Usage
```python
import joblib
import numpy as np

# Load trained model
model_data = joblib.load('house_price_model.pkl')

# Make prediction
features = np.array([[2000, 3, 2, 10, 2, 1, 1]])  # [area, bed, bath, age, garage, location, condition]
features_scaled = model_data['scaler'].transform(features)
price = model_data['model'].predict(features_scaled)[0]

print(f"Predicted price: ${price:,.2f}")
```

### Web Interface
1. Enter house details in the form
2. Click "Predict Price"
3. View detailed analysis and feature impact
4. Explore different scenarios

## Sample Predictions

| House Details | Predicted Price | Actual Price | Error |
|---------------|----------------|--------------|-------|
| 1673 sq ft, 1 bed, 3 bath, 24 years | $499,016 | $531,387 | 6.1% |
| 2057 sq ft, 3 bed, 3 bath, 37 years | $589,493 | $571,804 | 3.1% |
| 2450 sq ft, 3 bed, 2 bath, 22 years | $634,181 | $655,152 | 3.2% |

## Future Enhancements

- [ ] **Advanced Models**: Random Forest, XGBoost, Neural Networks
- [ ] **Real Dataset**: Integration with actual housing data APIs
- [ ] **Feature Engineering**: Derived features and polynomial terms
- [ ] **Cross-Validation**: K-fold validation for robust evaluation
- [ ] **Hyperparameter Tuning**: Grid search optimization
- [ ] **API Development**: REST API for external integrations
- [ ] **Mobile App**: React Native or Flutter application
- [ ] **Real-time Data**: Live market data integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Scikit-learn** for machine learning algorithms
- **Streamlit** for the amazing web framework
- **Netlify** for seamless deployment
- **Matplotlib/Seaborn** for beautiful visualizations

## Contact
through email: shivashankarbejjanki1@example.com

--------------------------------thanks for reading-----------------------------------
