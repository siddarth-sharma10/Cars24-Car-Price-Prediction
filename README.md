Cars24 Car Price Prediction
A complete end-to-end data science project that predicts the resale price of used cars listed on the Cars24 platform using machine learning.

📌 Project Overview
The used car market in India is growing rapidly. Platforms like Cars24 need accurate pricing to maximize revenue, build customer trust, and improve inventory turnover. This project builds a machine learning model that predicts a car's resale price based on its features like brand, model, fuel type, kilometers driven, transmission, and more.

Dataset : 6,437 Cars24 listings with 19 features
Target Variable : Resale Price (₹)
Best Model : XGBoost (Tuned) — R² = 0.9228, MAE = ₹69,727


📁 Project Structure
cars24_app/
│
├── Cars24_eda,model building.ipynb   # Complete EDA + Model Building notebook
├── app.py                        # Streamlit web application
├── train_and_save.py             # Script to train and save the model
├── cars24_data.csv               # Dataset
├── xgb_model.pkl                 # Saved trained model (generated after running train_and_save.py)
└── model_columns.pkl             # Saved model columns (generated after running train_and_save.py)

🔍 Workflow
Raw Data → EDA → Cleaning → Feature Engineering → Encoding → Model Training → Evaluation → Deployment

📊 EDA Highlights
FindingDetailDataset shape6,437 rows × 19 columnsMissing valuesOnly transmission (5.8%) and bodytype (5.8%)Price range₹1.34L — ₹30.48LPrice distributionRight-skewed (mean ₹6.78L, median ₹5.88L)Most common brandMaruti (40% of listings)Most common fuelPetrol (78%)Most common transmissionManual (77%)Most common body typeHatchback (55%)
Key Insights from EDA

Newer cars command significantly higher prices — car age has -0.51 correlation with price
Diesel cars are on average ₹3.5L more expensive than petrol
Automatic cars command a ₹3.2L premium over manual
1st owner cars retain value far better than 2nd or 3rd owner
SUVs and Luxury SUVs have the highest average resale value (₹10L+)
Higher mileage reduces price — -0.20 correlation with price


⚙️ Feature Engineering
FeatureDescriptioncar_age2024 − manufacturing yearbrand_tierLuxury / Premium / Economy groupingkm_per_yearkilometerdriven ÷ car_ageis_first_owner1 if 1st owner, else 0age_km_ratiocar_age × kilometerdriven / 1000log_kmlog1p of kilometers drivenmodel_cleanTop 20 car models kept, rest grouped as 'Other'

🤖 Models Trained & Results
ModelR² ScoreMAELinear Regression0.7930₹1,13,811Decision Tree0.7573₹1,18,565Random Forest0.8466₹94,577XGBoost (original)0.8612₹91,606XGBoost (tuned)0.9228₹69,727
What pushed R² from 0.86 → 0.92

model_clean — added car model (Swift, Creta, Dzire etc.) as a feature. Biggest single improvement.
Extra features — km_per_year, is_first_owner, age_km_ratio, log_km
Tuned XGBoost hyperparameters via RandomizedSearchCV (30 iterations, 3-fold CV)

Best XGBoost Parameters
pythonXGBRegressor(
    n_estimators     = 300,
    learning_rate    = 0.03,
    max_depth        = 8,
    subsample        = 0.9,
    colsample_bytree = 0.75,
    min_child_weight = 3,
    reg_alpha        = 0,
    reg_lambda       = 2,
    random_state     = 42
)

🌐 Streamlit Web App
The trained model is deployed as an interactive web application built with Streamlit.
Features of the app:

Select brand → car model dropdown automatically filters to that brand's cars only
Select a car model → body type auto-fills (Creta = SUV, Dzire = Sedan etc.)
Shows predicted price + likely price range (±8%)
Displays key factors influencing the price (car age, km, brand tier, ownership)
Depreciation guide expander
Cars24 Assured toggle


🚀 How to Run
Step 1 — Clone the repository
bashgit clone https://github.com/yourusername/cars24-price-prediction.git
cd cars24-price-prediction
Step 2 — Install dependencies
bashpip install -r requirements.txt
Step 3 — Train and save the model (run once)
bashpython train_and_save.py
This generates xgb_model.pkl and model_columns.pkl
Step 4 — Run the web app
bashstreamlit run app.py
App opens at http://localhost:8501

📦 Requirements
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
seaborn
scipy

📈 Tech Stack
ToolPurposePythonCore languagePandas & NumPyData manipulationMatplotlib & SeabornVisualisationScikit-learnPreprocessing & ML modelsXGBoostBest performing modelJoblibModel serialisationStreamlitWeb app deployment

👤 Author
Siddarth Sharma
Data Science Intern
April 2026
