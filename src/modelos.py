"""Clasificadores propios. Viven aqui para que la app y reentrenar.py puedan cargar el modelo guardado."""
from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class XGBBalanceado(ClassifierMixin, BaseEstimator):
    """XGBoost con pesos por clase: pide etiquetas numericas y no tiene class_weight, asi que se codifican y se pasan pesos."""

    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=6, min_child_weight=1, subsample=1.0,
                 colsample_bytree=1.0, reg_lambda=1.0, n_jobs=-1, random_state=42):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.min_child_weight, self.subsample, self.colsample_bytree = min_child_weight, subsample, colsample_bytree
        self.reg_lambda, self.n_jobs, self.random_state = reg_lambda, n_jobs, random_state

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.m_ = XGBClassifier(**self.get_params(), tree_method="hist", verbosity=0)
        self.m_.fit(X, self.le_.transform(y), sample_weight=compute_sample_weight("balanced", y))
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.m_.predict(X))

    def predict_proba(self, X):
        return self.m_.predict_proba(X)
