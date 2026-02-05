# House Price Prediction

This project predicts house prices using feature engineering and regression models based on the Kaggle House Prices dataset.

## What the project does
- Handles missing values systematically  
- Creates new features:
  - TotalSF  
  - HouseAge  
  - RemodAge  
- One-hot encodes categorical variables  
- Splits data into train/test  
- Trains and compares multiple models  

---

## Models compared
- Linear Regression  
- Ridge Regression (with multiple alpha values)  
- Gradient Boosting Regressor  

Evaluation metric: RMSE (Root Mean Squared Error).  
Final model selected: Ridge Regression (alpha = 5).

---

## Results (Kaggle)

Best public leaderboard score: **9.46024**  
Kaggle competition: *House Prices – Advanced Regression Techniques*

This score was achieved using Ridge Regression (alpha = 5) after systematic feature engineering and missing-value handling.

---

## How to run
```bash
pip install -r requirements.txt
python house_prices.py

---

This will output:

- submissions/submission_ridge.csv 
