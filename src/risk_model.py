import pandas as pd 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv("data/infrarisk_dataset.csv")

X = df.drop("Default", axis=1)
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42    
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
lgb_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

rf_model.fit(X_train,y_train)
xgb_model.fit(X_train,y_train)
lgb_model.fit(X_train,y_train)

rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))

print(f"RandomForest Accuracy : {rf_acc*100:.2f}%")
print(f"XGBoost Accuracy      : {xgb_acc*100:.2f}%")
print(f"LightGBM Accuracy     : {lgb_acc*100:.2f}%")

joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(lgb_model, "models/lgb_model.pkl")

print("All Models Saved Successfully")