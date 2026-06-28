# ==========================================================
# MOVIE RATING PREDICTION USING RANDOM FOREST REGRESSION
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# STEP 1 : Load Dataset
# ==========================================================

df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

print("="*60)
print("First Five Rows")
print("="*60)
print(df.head())

print("\nDataset Shape :", df.shape)

# ==========================================================
# STEP 2 : Dataset Information
# ==========================================================

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================================
# STEP 3 : Handle Missing Values
# ==========================================================

# Fill categorical columns
categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

for col in categorical_cols:
    if col in df.columns:
        df[col].fillna("Unknown", inplace=True)

# Fill numeric columns
if 'Duration' in df.columns:
    df['Duration'] = df['Duration'].astype(str).str.extract('(\d+)')
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    df['Duration'].fillna(df['Duration'].median(), inplace=True)

if 'Votes' in df.columns:
    df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    df['Votes'].fillna(df['Votes'].median(), inplace=True)

# Remove rows without Rating
df.dropna(subset=['Rating'], inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ==========================================================
# STEP 4 : Encode Categorical Features
# ==========================================================

encoder = LabelEncoder()

for col in categorical_cols:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])

# ==========================================================
# STEP 5 : Process Year
# ==========================================================

if 'Year' in df.columns:
    df['Year'] = df['Year'].astype(str).str.extract('(\d{4})')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Year'].fillna(df['Year'].median(), inplace=True)

# ==========================================================
# STEP 6 : Remove Unnecessary Columns
# ==========================================================

if 'Name' in df.columns:
    df.drop('Name', axis=1, inplace=True)

# ==========================================================
# STEP 7 : Correlation Heatmap
# ==========================================================

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# STEP 8 : Feature Selection
# ==========================================================

X = df.drop('Rating', axis=1)

y = df['Rating']

# ==========================================================
# STEP 9 : Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# STEP 10 : Train Model
# ==========================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# STEP 11 : Prediction
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# STEP 12 : Evaluation
# ==========================================================

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error :", round(mae,3))
print("Mean Squared Error :", round(mse,3))
print("Root Mean Squared Error :", round(rmse,3))
print("R2 Score :", round(r2,3))

# ==========================================================
# STEP 13 : Actual vs Predicted Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Rating")

plt.ylabel("Predicted Rating")

plt.title("Actual vs Predicted Ratings")

plt.grid(True)

plt.show()

# ==========================================================
# STEP 14 : Feature Importance
# ==========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance")

plt.show()

# ==========================================================
# STEP 15 : Predict Rating for New Movie
# ==========================================================

print("\nPrediction for Sample Movie")

sample_movie = pd.DataFrame({
    'Year':[2022],
    'Duration':[140],
    'Genre':[10],
    'Director':[25],
    'Actor 1':[100],
    'Actor 2':[110],
    'Actor 3':[150],
    'Votes':[50000]
})

predicted_rating = model.predict(sample_movie)

print("Predicted Rating :", round(predicted_rating[0],2))

# ==========================================================
# STEP 16 : Save Model
# ==========================================================

pickle.dump(model, open("movie_rating_model.pkl","wb"))

print("\nModel Saved Successfully")

print("\nProject Completed Successfully")