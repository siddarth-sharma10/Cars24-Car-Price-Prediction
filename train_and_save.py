"""
Run this file ONCE to train the model and save it.
Command: python train_and_save.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

print("Loading data...")
df = pd.read_csv('cars24_data.csv')

# ── same cleaning as EDA ──
df_clean = df.drop(columns=['url', 'storename', 'name', 'registrationcity'])
df_clean['bodytype'] = df_clean['bodytype'].str.strip().str.title()
df_clean['createdDate'] = pd.to_datetime(df_clean['createdDate'], dayfirst=True, errors='coerce')
df_clean['listing_month'] = df_clean['createdDate'].dt.month
df_clean['listing_year']  = df_clean['createdDate'].dt.year
for col in ['transmission', 'bodytype']:
    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

CURRENT_YEAR = 2024
df_clean['car_age']     = CURRENT_YEAR - df_clean['year']
df_clean['price_per_km']= (df_clean['price'] / df_clean['kilometerdriven']).round(2)

luxury_brands  = ['Audi','Bmw','Mercedes Benz','Jaguar','Porsche','Volvo','Land Rover']
premium_brands = ['Mg','Jeep','Kia','Skoda','Volkswagen','Toyota']
df_clean['brand_tier']  = df_clean['make'].apply(
    lambda m: 'Luxury' if m in luxury_brands else ('Premium' if m in premium_brands else 'Economy'))
q75 = df_clean['kilometerdriven'].quantile(0.75)
df_clean['high_mileage']= (df_clean['kilometerdriven'] > q75).astype(int)
df_clean['km_bucket']   = pd.cut(df_clean['kilometerdriven'],
    bins=[0,20000,40000,60000,80000,100000,500000],
    labels=['0-20K','20-40K','40-60K','60-80K','80-100K','100K+'])

# outlier removal
Q1_p=df_clean['price'].quantile(0.25); Q3_p=df_clean['price'].quantile(0.75)
price_upper = Q3_p + 3*(Q3_p-Q1_p)
Q1_k=df_clean['kilometerdriven'].quantile(0.25); Q3_k=df_clean['kilometerdriven'].quantile(0.75)
km_upper = Q3_k + 3*(Q3_k-Q1_k)
df_no_outliers = df_clean[(df_clean['price']<=price_upper)&(df_clean['kilometerdriven']<=km_upper)].copy()

# ── extra features (v2) ──
df_v2 = df_no_outliers.copy()
df_v2['km_per_year']    = (df_v2['kilometerdriven'] / df_v2['car_age'].replace(0, 1)).round(2)
df_v2['is_first_owner'] = (df_v2['ownernumber'] == 1).astype(int)
df_v2['age_km_ratio']   = (df_v2['car_age'] * df_v2['kilometerdriven'] / 1000).round(2)
df_v2['log_km']         = np.log1p(df_v2['kilometerdriven'])

top_models = df_v2['model'].value_counts().head(20).index
df_v2['model_clean'] = df_v2['model'].where(df_v2['model'].isin(top_models), 'Other')

cols_to_drop = ['year','createdDate','listing_month','listing_year',
                'price_per_km','registrationstate','model','km_bucket']
df_v2 = df_v2.drop(columns=cols_to_drop, errors='ignore').copy()

le = LabelEncoder()
df_v2['transmission'] = le.fit_transform(df_v2['transmission'].astype(str))
df_v2['isc24assured'] = df_v2['isc24assured'].astype(int)
df_v2['high_mileage'] = df_v2['high_mileage'].astype(int)

cat_cols = ['make','city','fueltype','bodytype','brand_tier','model_clean']
df_v2 = pd.get_dummies(df_v2, columns=cat_cols, drop_first=True)
bool_cols = df_v2.select_dtypes(include='bool').columns
df_v2[bool_cols] = df_v2[bool_cols].astype(int)

df_v2['log_price'] = np.log1p(df_v2['price'])

X = df_v2.drop(columns=['price','log_price'])
y = df_v2['log_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features...")

# ── train best model ──
model = XGBRegressor(
    n_estimators     = 300,
    learning_rate    = 0.03,
    max_depth        = 8,
    subsample        = 0.9,
    colsample_bytree = 0.75,
    min_child_weight = 3,
    reg_alpha        = 0,
    reg_lambda       = 2,
    random_state     = 42,
    verbosity        = 0
)
model.fit(X_train, y_train)

# ── save model + column order ──
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

print("✅  Model saved as  xgb_model.pkl")
print("✅  Columns saved as model_columns.pkl")
print(f"    Features used : {X_train.shape[1]}")
