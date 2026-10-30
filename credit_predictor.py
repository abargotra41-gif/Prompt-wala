"""
credit_predictor.py
Machine Learning Credit Period Predictor and Policy Guardrails for Kirana Stores.
Evaluates both Logistic Regression and Decision Tree models with dummy variables
on 300 Kirana store profiles (precision, recall, F1-score, accuracy).
Enforces two mandatory credit guardrails:
1. Overdue Balance Guardrail: Overdue balance > ₹10,000 strictly blocks credit period.
2. Transaction Value Guardrail: Order transaction total > ₹30,000 strictly blocks credit (must be COD).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


@dataclass
class ModelMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass
class CreditAssessment:
    store_id: str
    store_name: str
    city: str
    store_type: str
    order_value_inr: float
    overdue_balance_inr: float
    credit_limit_inr: float
    on_time_payment_rate: float
    average_days_late: float
    has_established_history: bool
    # Guardrail 1: Overdue > ₹10,000
    exceeds_10k_threshold: bool
    # Guardrail 2: Order value > ₹30,000
    exceeds_30k_order_threshold: bool
    guardrail_blocked: bool
    guardrail_message: str
    # ML Prediction
    model_prediction: int  # 1 = Credit Period, 0 = No Credit
    model_credit_granted: bool
    model_probability: float
    best_model_name: str
    recommended_credit_days: int  # 0, 7, or 14
    recommended_payment_terms: str
    explanation: str
    poor_payment_history_blocked: bool = False


class CreditPeriodPredictor:
    def __init__(self, data_path: Optional[str] = None):
        default_path = os.path.join(
            os.path.dirname(__file__),
            "updated guard rails.xlsx",
        )
        if not os.path.exists(default_path):
            default_path = os.path.join(
                os.path.dirname(__file__),
                "Synthetic_Kirana_AI_Dataset_300_Stores_Credit_Guardrails.xlsx",
            )
        self.data_path = data_path or default_path
        self.df_stores: pd.DataFrame = pd.DataFrame()
        self.feature_columns: List[str] = []
        self.lr_metrics: Optional[ModelMetrics] = None
        self.dt_metrics: Optional[ModelMetrics] = None
        self.best_model: Any = None
        self.best_model_name: str = "Logistic Regression"
        self._load_and_train()

    def _load_and_train(self) -> None:
        """Load 300 stores dataset, prepare dummy variables, train LR & DT, and pick best model."""
        if os.path.exists(self.data_path):
            try:
                self.df_stores = pd.read_excel(self.data_path, sheet_name="Stores")
            except Exception:
                pass

        if self.df_stores.empty or "Credit_Period_Granted_By_Policy" not in self.df_stores.columns:
            csv_path = os.path.join(os.path.dirname(__file__), "data", "Stores.csv")
            if os.path.exists(csv_path):
                self.df_stores = pd.read_csv(csv_path)

        if self.df_stores.empty or "Credit_Period_Granted_By_Policy" not in self.df_stores.columns:
            return

        # Prepare target
        y = self.df_stores["Credit_Period_Granted_By_Policy"].astype(int)

        # Numeric features with clean imputation
        num_cols = [
            "On_Time_Payment_Rate",
            "Average_Days_Late",
            "Overdue_Balance_INR",
            "Credit_Limit_INR",
            "Has_Established_Payment_History",
        ]
        df_num = self.df_stores[num_cols].copy()
        df_num["On_Time_Payment_Rate"] = df_num["On_Time_Payment_Rate"].fillna(0.0)
        df_num["Average_Days_Late"] = df_num["Average_Days_Late"].fillna(30.0)
        df_num["Overdue_Balance_INR"] = df_num["Overdue_Balance_INR"].fillna(0.0)
        df_num["Credit_Limit_INR"] = df_num["Credit_Limit_INR"].fillna(0.0)

        # Categorical features encoded with dummy variables
        cat_cols = ["Payment_Behaviour_History", "Store_Type", "City"]
        df_cat = pd.get_dummies(self.df_stores[cat_cols], drop_first=True)

        X = pd.concat([df_num, df_cat], axis=1)
        self.feature_columns = list(X.columns)

        # Train/Test Split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # 1. Logistic Regression Model with Feature Scaling
        pipe_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)

        self.lr_metrics = ModelMetrics(
            model_name="Logistic Regression",
            accuracy=float(accuracy_score(y_test, y_pred_lr)),
            precision=float(precision_score(y_test, y_pred_lr, zero_division=0)),
            recall=float(recall_score(y_test, y_pred_lr, zero_division=0)),
            f1_score=float(f1_score(y_test, y_pred_lr, zero_division=0)),
            support=len(y_test),
        )

        # 2. Decision Tree Model (max_depth=4 to prevent overfitting)
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)

        self.dt_metrics = ModelMetrics(
            model_name="Decision Tree",
            accuracy=float(accuracy_score(y_test, y_pred_dt)),
            precision=float(precision_score(y_test, y_pred_dt, zero_division=0)),
            recall=float(recall_score(y_test, y_pred_dt, zero_division=0)),
            f1_score=float(f1_score(y_test, y_pred_dt, zero_division=0)),
            support=len(y_test),
        )

        # Pick best model based on F1-Score
        if self.lr_metrics.f1_score >= self.dt_metrics.f1_score:
            self.best_model = pipe_lr
            self.best_model_name = "Logistic Regression"
        else:
            self.best_model = dt
            self.best_model_name = "Decision Tree"

    def assess_store_credit(self, store_id: str, order_value: Decimal = Decimal("20000")) -> CreditAssessment:
        """
        Assess credit eligibility for a specific store.
        Enforces two policy guardrails:
        1. Order Value Guardrail: order_value > ₹30,000 -> credit strictly blocked (must be COD).
        2. Overdue Balance Guardrail: overdue > ₹10,000 -> credit strictly blocked.
        If both guardrails pass, predicts credit period using best ML model.
        """
        ov_float = float(order_value) if order_value is not None else 0.0

        if self.df_stores.empty:
            return CreditAssessment(
                store_id=store_id,
                store_name=store_id,
                city="Unknown",
                store_type="Kirana",
                order_value_inr=ov_float,
                overdue_balance_inr=0.0,
                credit_limit_inr=0.0,
                on_time_payment_rate=0.0,
                average_days_late=0.0,
                has_established_history=False,
                exceeds_10k_threshold=False,
                exceeds_30k_order_threshold=False,
                guardrail_blocked=False,
                guardrail_message="No store credit data available.",
                model_prediction=0,
                model_credit_granted=False,
                model_probability=0.0,
                best_model_name="N/A",
                recommended_credit_days=0,
                recommended_payment_terms="Cash on Delivery (COD)",
                explanation="No credit data found.",
            )

        match = self.df_stores[self.df_stores["Store_ID"] == store_id]
        if match.empty:
            row = self.df_stores.iloc[0]
            store_name = store_id
        else:
            row = match.iloc[0]
            store_name = str(row.get("Store_Name", store_id))

        city = str(row.get("City", "Unknown"))
        store_type = str(row.get("Store_Type", "General"))
        overdue = float(row.get("Overdue_Balance_INR", 0.0) or 0.0)
        credit_limit = float(row.get("Credit_Limit_INR", 0.0) or 0.0)
        on_time = float(row.get("On_Time_Payment_Rate", 0.0) or 0.0)
        days_late = float(row.get("Average_Days_Late", 0.0) or 0.0)
        has_history = bool(int(row.get("Has_Established_Payment_History", 1) or 1))

        # -------------------------------------------------------------
        # GUARDRAIL 1: Transaction Value Guardrail (₹30,000 Limit)
        # Transactions above ₹30,000 are not allowed for credit and must be COD
        # -------------------------------------------------------------
        exceeds_30k = ov_float > 30000.0
        if exceeds_30k:
            guardrail_msg = (
                f"🛑 POLICY GUARDRAIL (₹30,000 Transaction Limit): Order total is ₹{ov_float:,.2f}, "
                f"which exceeds the maximum transaction credit ceiling of ₹30,000.00. "
                f"Orders above ₹30,000 are not allowed for credit and must be processed via Cash on Delivery (COD) / Full Upfront Payment."
            )
            return CreditAssessment(
                store_id=store_id,
                store_name=store_name,
                city=city,
                store_type=store_type,
                order_value_inr=ov_float,
                overdue_balance_inr=overdue,
                credit_limit_inr=credit_limit,
                on_time_payment_rate=on_time,
                average_days_late=days_late,
                has_established_history=has_history,
                exceeds_10k_threshold=(overdue > 10000.0),
                exceeds_30k_order_threshold=True,
                guardrail_blocked=True,
                guardrail_message=guardrail_msg,
                model_prediction=0,
                model_credit_granted=False,
                model_probability=0.0,
                best_model_name=self.best_model_name,
                recommended_credit_days=0,
                recommended_payment_terms="COD / Full Upfront Payment Required",
                explanation=guardrail_msg,
            )

        # -------------------------------------------------------------
        # GUARDRAIL 2: Overdue Balance Guardrail (₹10,000 Limit)
        # Unpaid overdue balance > ₹10,000 strictly blocks credit
        # -------------------------------------------------------------
        exceeds_10k = overdue > 10000.0
        if exceeds_10k:
            guardrail_msg = (
                f"🛑 POLICY GUARDRAIL (₹10,000 Overdue Limit): This store has an unpaid overdue balance of ₹{overdue:,.2f}, "
                f"which exceeds the maximum allowable threshold of ₹10,000.00. "
                f"They must finish/clear the last credit that they are left with before another credit period can be granted."
            )
            return CreditAssessment(
                store_id=store_id,
                store_name=store_name,
                city=city,
                store_type=store_type,
                order_value_inr=ov_float,
                overdue_balance_inr=overdue,
                credit_limit_inr=credit_limit,
                on_time_payment_rate=on_time,
                average_days_late=days_late,
                has_established_history=has_history,
                exceeds_10k_threshold=True,
                exceeds_30k_order_threshold=False,
                guardrail_blocked=True,
                guardrail_message=guardrail_msg,
                model_prediction=0,
                model_credit_granted=False,
                model_probability=0.0,
                best_model_name=self.best_model_name,
                recommended_credit_days=0,
                recommended_payment_terms="COD / Full Upfront Payment Required",
                explanation=guardrail_msg,
            )

        # -------------------------------------------------------------
        # Model Prediction (when both guardrails pass)
        # -------------------------------------------------------------
        guardrail_msg = f"✅ Guardrails passed (Order ₹{ov_float:,.2f} <= ₹30k and Overdue ₹{overdue:,.2f} <= ₹10k)."

        # Build feature vector matching training columns
        feat_dict = {
            "On_Time_Payment_Rate": on_time,
            "Average_Days_Late": days_late,
            "Overdue_Balance_INR": overdue,
            "Credit_Limit_INR": credit_limit,
            "Has_Established_Payment_History": 1 if has_history else 0,
        }
        for col in self.feature_columns:
            if col not in feat_dict:
                feat_dict[col] = 0

        pbh = str(row.get("Payment_Behaviour_History", ""))
        stt = str(row.get("Store_Type", ""))
        cty = str(row.get("City", ""))
        if f"Payment_Behaviour_History_{pbh}" in feat_dict:
            feat_dict[f"Payment_Behaviour_History_{pbh}"] = 1
        if f"Store_Type_{stt}" in feat_dict:
            feat_dict[f"Store_Type_{stt}"] = 1
        if f"City_{cty}" in feat_dict:
            feat_dict[f"City_{cty}"] = 1

        X_single = pd.DataFrame([feat_dict])[self.feature_columns]

        prob = 0.5
        pred = 0
        if self.best_model is not None:
            pred = int(self.best_model.predict(X_single)[0])
            if hasattr(self.best_model, "predict_proba"):
                prob = float(self.best_model.predict_proba(X_single)[0][1])
            else:
                prob = 1.0 if pred == 1 else 0.0

        if pred == 1:
            credit_days = 14 if (on_time >= 0.90 and days_late <= 3.0 and overdue <= 500) else 7
            payment_terms = f"{credit_days}-Day Credit Period"
            explanation = (
                f"Approved for {credit_days}-day payment credit (Confidence: {prob*100:.0f}% by {self.best_model_name}). "
                f"On-time rate is {on_time*100:.0f}%, average delay is {days_late:.1f} days, and overdue balance is ₹{overdue:,.2f}."
            )
            credit_granted = True
            poor_payment_blocked = False
        else:
            credit_days = 0
            payment_terms = "Cash on Delivery (COD) / Prepaid Required"
            explanation = (
                f"Guardrail 3 Failed (ML Credit Risk): Model predicts elevated default risk based on past transaction history "
                f"(On-time rate: {on_time*100:.0f}%, delay: {days_late:.1f} days, behavior: '{pbh}'). "
                f"Credit period is denied by policy guardrail. Terms required: Cash on Delivery (COD) / Prepaid."
            )
            credit_granted = False
            poor_payment_blocked = True

        return CreditAssessment(
            store_id=store_id,
            store_name=store_name,
            city=city,
            store_type=store_type,
            order_value_inr=ov_float,
            overdue_balance_inr=overdue,
            credit_limit_inr=credit_limit,
            on_time_payment_rate=on_time,
            average_days_late=days_late,
            has_established_history=has_history,
            exceeds_10k_threshold=False,
            exceeds_30k_order_threshold=False,
            guardrail_blocked=poor_payment_blocked,
            guardrail_message=explanation if poor_payment_blocked else guardrail_msg,
            model_prediction=pred,
            model_credit_granted=credit_granted,
            model_probability=prob,
            best_model_name=self.best_model_name,
            recommended_credit_days=credit_days,
            recommended_payment_terms=payment_terms,
            explanation=explanation,
            poor_payment_history_blocked=poor_payment_blocked,
        )
