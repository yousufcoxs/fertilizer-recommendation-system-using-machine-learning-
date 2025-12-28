# Fertilizer Recommendation System

 
An intelligent web application that recommends the most suitable fertilizers for agricultural crops based on soil conditions, weather parameters, and crop type using machine learning (XGBoost model).

## Features

- 🌱 Crop-specific fertilizer recommendations
- 🌦️ Considers temperature, moisture, rainfall, and soil pH
- 🧪 Analyzes nitrogen, phosphorus, potassium, and carbon levels
- 🖥️ User-friendly web interface
- 🤖 Powered by XGBoost machine learning model
- 📊 Dataset includes rice, wheat, maize, mung bean, millet, and tea crops

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript (possibly React.js or Flask templates)
- **Backend**: Python (Flask/Django)
- **Machine Learning**: 
  - XGBoost model
  - Scikit-learn for preprocessing
  - Pandas for data handling
- **Dataset**: Includes soil parameters, weather conditions, and crop types

## Dataset

The system uses a comprehensive dataset containing:
- Temperature
- Moisture levels
- Rainfall
- Soil pH
- Nitrogen, Phosphorous, Potassium levels
- Carbon content
- Soil type (Loamy, Peaty, Acidic, Neutral)
- Crop type (rice, wheat, maize, mung bean, millet, tea)
- Recommended fertilizers with explanations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fertilizer-recommendation-system.git
