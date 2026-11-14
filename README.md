# ☀️ SolarPump AI

AI-powered next-day solar energy forecasting and smart irrigation recommendation system for agricultural applications.

---

## 🚀 Overview
SolarPump AI predicts next-day solar energy generation using meteorological and solar data. By leveraging machine learning, the application provides actionable, 3-level irrigation recommendations (**HIGH**, **MODERATE**, **LOW**) to help farmers optimize water pump usage based on real-time weather conditions or live forecasts.

---

## 📊 Model Evaluation & Performance
The project compares a baseline model against linear and non-linear machine learning regressors using a chronological 2025 holdout evaluation dataset:

* **Persistence Baseline:** MAE = 1.966
* **Linear Regression:** MAE = 1.896
* **Random Forest Regressor:** **MAE = 1.865** *(Best Performing Model)*
* **Tier Classification Accuracy:** **92.3%**

---

## 🌦️ Key Features
* **Live Weather Integration:** Automatically fetches current meteorological forecasts via the Open-Meteo API.
* **Manual Testing Sliders:** Allows users to adjust weather variables manually to test different environmental scenarios.
* **Smart Irrigation Recommendations:** Categorizes energy outputs into High, Moderate, and Low tiers to guide safe and efficient pump operation.
* **Interactive Web Interface:** Powered by Gradio for an intuitive browser-based user experience.

---

## ⚙️ Input Features
The prediction model utilizes the following daily meteorological parameters:
* **Solar Radiation** (Shortwave radiation sum)
* **Clear-Sky Radiation**
* **Maximum Temperature**
* **Relative Humidity**

---

## 📂 Project Structure
```text
solar-pump-ai/
│
├── app.py                  # Gradio web application interface
├── train_model.py          # Data download, model training, and evaluation script
├── requirements.txt        # Python package dependencies
└── solar_pump_ai.ipynb     # Google Colab launcher notebook