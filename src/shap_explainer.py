import joblib
import shap
import pandas as pd

model = joblib.load("models/xgb_model.pkl")

explainer = shap.TreeExplainer(model)

print("SHAP Explainer Ready")