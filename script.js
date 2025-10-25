// House Price Prediction JavaScript

// Model coefficients (from the trained Linear Regression model)
const MODEL_COEFFICIENTS = {
    area: 72394.74,
    bedrooms: 29241.24,
    bathrooms: 13402.21,
    age: -30132.15,
    garage: 7278.73,
    location_encoded: -12299.16,
    condition_encoded: -12335.99
};

const BASE_PRICE = 100000;

// Location encoding
const LOCATION_ENCODING = {
    'Downtown': 2,
    'Suburb': 1,
    'Rural': 0
};

// Condition encoding
const CONDITION_ENCODING = {
    'Excellent': 0,
    'Good': 1,
    'Fair': 2,
    'Poor': 3
};

// Location premiums
const LOCATION_PREMIUMS = {
    'Downtown': 50000,
    'Suburb': 20000,
    'Rural': 0
};

// Condition premiums
const CONDITION_PREMIUMS = {
    'Excellent': 30000,
    'Good': 15000,
    'Fair': 5000,
    'Poor': -10000
};

function updateAgeValue(value) {
    document.getElementById('ageValue').textContent = `${value} years`;
}

function predictPrice(area, bedrooms, bathrooms, age, garage, location, condition) {
    // Encode categorical variables
    const locationEncoded = LOCATION_ENCODING[location];
    const conditionEncoded = CONDITION_ENCODING[condition];
    
    // Calculate base price using model coefficients (simplified)
    let price = BASE_PRICE;
    
    // Add feature contributions
    price += area * 150; // Simplified coefficient
    price += bedrooms * 20000;
    price += bathrooms * 15000;
    price += (50 - age) * 2000; // Newer houses are more valuable
    price += garage * 10000;
    price += LOCATION_PREMIUMS[location];
    price += CONDITION_PREMIUMS[condition];
    
    // Add some realistic variation
    const variation = (Math.random() - 0.5) * 40000;
    price += variation;
    
    // Ensure minimum price
    return Math.max(price, 50000);
}

function calculateMetrics(price, area, bedrooms) {
    return {
        pricePerSqft: price / area,
        pricePerBedroom: bedrooms > 0 ? price / bedrooms : 0,
        marketSegment: price < 200000 ? 'Budget' : price < 400000 ? 'Mid-Range' : 'Premium'
    };
}

function calculateFeatureImpact(area, bedrooms, bathrooms, age, garage, location, condition) {
    return {
        'Area': area * 150,
        'Bedrooms': bedrooms * 20000,
        'Bathrooms': bathrooms * 15000,
        'Newness': (50 - age) * 2000,
        'Garage': garage * 10000,
        'Location': LOCATION_PREMIUMS[location],
        'Condition': CONDITION_PREMIUMS[condition]
    };
}

function displayFeatureImpact(impacts) {
    const impactBars = document.getElementById('impactBars');
    impactBars.innerHTML = '';
    
    // Sort impacts by absolute value
    const sortedImpacts = Object.entries(impacts)
        .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a));
    
    const maxImpact = Math.max(...Object.values(impacts).map(Math.abs));
    
    sortedImpacts.forEach(([feature, impact]) => {
        const barContainer = document.createElement('div');
        barContainer.className = 'impact-bar';
        
        const label = document.createElement('div');
        label.className = 'impact-label';
        label.textContent = feature;
        
        const visual = document.createElement('div');
        visual.className = 'impact-visual';
        const width = Math.abs(impact) / maxImpact * 100;
        visual.style.width = `${width}%`;
        visual.style.background = impact >= 0 ? 
            'linear-gradient(90deg, #2e8b57, #90ee90)' : 
            'linear-gradient(90deg, #dc143c, #ffb6c1)';
        
        const value = document.createElement('div');
        value.className = 'impact-value';
        value.textContent = `${impact >= 0 ? '+' : ''}$${impact.toLocaleString()}`;
        
        barContainer.appendChild(label);
        barContainer.appendChild(visual);
        barContainer.appendChild(value);
        
        impactBars.appendChild(barContainer);
    });
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

function handleFormSubmit(event) {
    event.preventDefault();
    
    // Add loading state
    document.body.classList.add('loading');
    
    // Get form values
    const area = parseFloat(document.getElementById('area').value);
    const bedrooms = parseInt(document.getElementById('bedrooms').value);
    const bathrooms = parseInt(document.getElementById('bathrooms').value);
    const age = parseInt(document.getElementById('age').value);
    const garage = parseInt(document.getElementById('garage').value);
    const location = document.getElementById('location').value;
    const condition = document.getElementById('condition').value;
    
    // Simulate processing delay
    setTimeout(() => {
        // Calculate prediction
        const predictedPrice = predictPrice(area, bedrooms, bathrooms, age, garage, location, condition);
        
        // Calculate metrics
        const metrics = calculateMetrics(predictedPrice, area, bedrooms);
        
        // Calculate feature impacts
        const impacts = calculateFeatureImpact(area, bedrooms, bathrooms, age, garage, location, condition);
        
        // Display results
        document.getElementById('predictedPrice').textContent = formatCurrency(predictedPrice);
        document.getElementById('pricePerSqft').textContent = formatCurrency(metrics.pricePerSqft);
        document.getElementById('pricePerBedroom').textContent = formatCurrency(metrics.pricePerBedroom);
        document.getElementById('marketSegment').textContent = metrics.marketSegment;
        
        // Display feature impact
        displayFeatureImpact(impacts);
        
        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
        
        // Remove loading state
        document.body.classList.remove('loading');
    }, 1000);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Add form submit handler
    document.getElementById('predictionForm').addEventListener('submit', handleFormSubmit);
    
    // Initialize age slider
    updateAgeValue(document.getElementById('age').value);
    
    console.log('House Price Predictor initialized successfully!');
});
