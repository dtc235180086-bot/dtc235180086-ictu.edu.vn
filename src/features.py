"""Đặc trưng và pipeline dùng chung cho huấn luyện, đánh giá và ứng dụng dự đoán.

Tách thành module riêng để joblib nạp lại được mô hình đã lưu.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUM = ["age", "bmi", "children"]
BIN = ["sex", "smoker"]
CAT = ["region"]
FEATURES = NUM + BIN + CAT
ENGINEERED = ["bmi_obese", "smoker_obese", "smoker_bmi", "age2"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    smoker = (X["smoker"] == "yes").astype(float)
    X["bmi_obese"] = (X["bmi"] >= 30).astype(float)
    # Chi phí người hút thuốc tăng vọt khi BMI ≥ 30 (mục 1.7) → đặc trưng tương tác
    X["smoker_obese"] = smoker * X["bmi_obese"]
    X["smoker_bmi"] = smoker * X["bmi"]
    X["age2"] = X["age"] ** 2
    return X


def preprocessor(engineered: bool = False) -> ColumnTransformer:
    num = NUM + (ENGINEERED if engineered else [])
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("bin", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))]), BIN),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(drop="first", handle_unknown="ignore"))]), CAT),
    ], verbose_feature_names_out=False)


def make_pipeline(model, engineered: bool = False) -> Pipeline:
    steps = []
    if engineered:
        steps.append(("fe", FunctionTransformer(add_features)))
    steps += [("prep", preprocessor(engineered)), ("model", model)]
    return Pipeline(steps)


def log1p_inv(y):
    return np.expm1(y)
