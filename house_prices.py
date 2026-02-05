import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


train = pd.read_csv(r"C:\coding\MACHINE\house-prices-advanced-regression-techniques\train_house.csv")
test = pd.read_csv(r"C:\coding\MACHINE\house-prices-advanced-regression-techniques\test_house.csv")
train_id = train['Id']
test_id = test['Id']

y = train["SalePrice"]
x = train.drop("SalePrice", axis=1)
y = np.log1p(y)
all_data = pd.concat([x, test], axis = 0)
none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu", 
             "GarageType", "GarageFinish", "GarageQual", "GarageCond",
             "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1"]

all_data[none_cols] = all_data[none_cols].fillna("None")

all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].median())
mode_cols = [
    "Functional", "SaleType", "Electrical", 
    "Exterior2nd", "KitchenQual", "Exterior1st", "MSZoning"
]
all_data[mode_cols] = all_data[mode_cols].fillna(all_data[mode_cols].mode().iloc[0])

zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
             'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea', 'MasVnrArea', 'GarageYrBlt']
all_data[zero_cols] = all_data[zero_cols].fillna(0)
all_data = all_data.drop(['Utilities'], axis=1)
all_data["TotalSF"] = ( all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"])
all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]
all_data = pd.get_dummies(all_data, drop_first= True)
x = all_data.iloc[:len(train), :]
test = all_data.iloc[len(train):, :]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
ridge = Ridge(alpha=10)
ridge.fit(x_train, y_train)
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gbr.fit(x_train, y_train)
scores = cross_val_score(
    ridge, x, y,
    cv=5,
    scoring="neg_mean_squared_error"
)
rmse_cv = np.sqrt(-scores.mean())
pred_linear = lr.predict(x_test)
rmse_linear = np.sqrt(mean_squared_error(y_test, pred_linear))
pred_gbr = gbr.predict(x_test)
rmse_gbr = np.sqrt(mean_squared_error(y_test, pred_gbr))
pred_ridge = ridge.predict(x_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
print("Linear RMSE:", rmse_linear)
print("Ridge RMSE:", rmse_ridge)
print("GB RMSE:", rmse_gbr)
print("CV RMSE:", rmse_cv)
for a in [0.1, 1, 5, 10, 20, 50, 100]:
    ridge = Ridge(alpha=a)
    ridge.fit(x_train, y_train)
    pred = ridge.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"alpha={a}, RMSE={rmse}")
ridge = Ridge(alpha=5)
ridge.fit(x_train, y_train)
best_model_pred = ridge.predict(test)
submission_ridge = pd.DataFrame({
    "Id": test_id,
    "SalePrice": best_model_pred
})
submission_ridge.to_csv("submission_ridge.csv", index=False)
print("Saved")