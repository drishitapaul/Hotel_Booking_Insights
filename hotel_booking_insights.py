"""
Hotel Booking Insights
IBM SkillsBuild Data Analytics with AI Internship Project

Run:
    pip install -r requirements.txt
    streamlit run hotel_booking_insights.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Hotel Booking Insights",
    page_icon="🏨",
    layout="wide"
)

DATA_PATH = Path(__file__).with_name("hotel_bookings.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Normalize column names without changing the dataset meaning.
    df.columns = [c.strip() for c in df.columns]

    # Basic cleaning
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")
    if "agent" in df.columns:
        df["agent"] = df["agent"].fillna(0)
    if "company" in df.columns:
        df["company"] = df["company"].fillna(0)

    # Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    # Useful derived fields for analytics.
    month_order = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    if "arrival_date_month" in df.columns:
        df["arrival_month_num"] = df["arrival_date_month"].map(month_order)

    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
        df["total_stay_nights"] = (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )

    if {"adults", "children", "babies"}.issubset(df.columns):
        df["total_guests"] = (
            df["adults"] + df["children"].fillna(0) + df["babies"]
        )

    if {"adr", "total_stay_nights"}.issubset(df.columns):
        df["estimated_booking_value"] = (
            df["adr"] * df["total_stay_nights"]
        )

    return df


@st.cache_resource
def train_models(df):
    # Predict cancellation using information that is available at/around booking
    # time. Outcome-revealing fields such as reservation_status and
    # reservation_status_date are deliberately excluded to avoid target leakage.
    target = "is_canceled"

    features = [
        "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
        "arrival_date_week_number", "arrival_date_day_of_month",
        "stays_in_weekend_nights", "stays_in_week_nights", "adults",
        "children", "babies", "meal", "country", "market_segment",
        "distribution_channel", "is_repeated_guest",
        "previous_cancellations", "previous_bookings_not_canceled",
        "reserved_room_type", "assigned_room_type", "booking_changes",
        "deposit_type", "days_in_waiting_list", "customer_type",
        "adr", "required_car_parking_spaces", "total_of_special_requests"
    ]

    available = [c for c in features if c in df.columns]
    model_df = df[available + [target]].copy()
    model_df = model_df[model_df[target].notna()].copy()

    X = model_df[available]
    y = model_df[target].astype(int)

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    log_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_depth=18
        ))
    ])

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    results = {}
    for name, model in [("Logistic Regression", log_model), ("Random Forest", rf_model)]:
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "pred": pred,
            "prob": prob,
            "cm": confusion_matrix(y_test, pred)
        }

    comparison = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [v["accuracy"] for v in results.values()],
        "Precision": [v["precision"] for v in results.values()],
        "Recall": [v["recall"] for v in results.values()],
        "F1 Score": [v["f1"] for v in results.values()],
        "ROC-AUC": [v["roc_auc"] for v in results.values()]
    })

    best_name = comparison.sort_values(
        ["ROC-AUC", "F1 Score"], ascending=False
    ).iloc[0]["Model"]

    return results, comparison, best_name, X_test, y_test


@st.cache_data
def monthly_bookings(df):
    if "arrival_date_month" not in df.columns:
        return pd.DataFrame()
    x = df.groupby(
        ["arrival_date_year", "arrival_month_num", "arrival_date_month"],
        as_index=False
    ).size()
    return x.sort_values(["arrival_date_year", "arrival_month_num"])


def main():
    st.title("🏨 Hotel Booking Insights")
    st.caption("Data Analytics & Cancellation Prediction")

    if not DATA_PATH.exists():
        st.error("hotel_bookings.csv was not found beside this Python file.")
        st.stop()

    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_hotel = st.sidebar.multiselect(
        "Hotel type",
        sorted(df["hotel"].dropna().unique()) if "hotel" in df else [],
        default=sorted(df["hotel"].dropna().unique()) if "hotel" in df else []
    )
    filtered = df[df["hotel"].isin(selected_hotel)].copy() if selected_hotel else df.copy()

    st.sidebar.caption(f"Rows after filters: {len(filtered):,}")

    # Executive overview
    st.header("1. Executive Overview")
    total = len(filtered)
    cancellation_rate = filtered["is_canceled"].mean() * 100
    avg_adr = filtered["adr"].mean()
    avg_stay = filtered["total_stay_nights"].mean()
    booking_value = filtered["estimated_booking_value"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bookings", f"{total:,}")
    c2.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
    c3.metric("Average ADR", f"€{avg_adr:,.2f}")
    c4.metric("Average Stay", f"{avg_stay:.1f} nights")
    c5.metric("Estimated Booking Value", f"€{booking_value:,.0f}")

    st.info(
        "Business lens: understand booking demand and cancellation patterns, "
        "then use a prediction model to identify reservations associated with higher cancellation risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        m = monthly_bookings(filtered)
        if not m.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = m["arrival_date_month"] + " " + m["arrival_date_year"].astype(str)
            ax.plot(labels, m["size"], marker="o")
            ax.set_title("Booking Demand by Arrival Month")
            ax.set_ylabel("Bookings")
            ax.tick_params(axis="x", rotation=75)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        if "hotel" in filtered.columns:
            hotel_cancel = filtered.groupby("hotel")["is_canceled"].mean().mul(100)
            fig, ax = plt.subplots(figsize=(8, 4))
            hotel_cancel.plot(kind="bar", ax=ax)
            ax.set_title("Cancellation Rate by Hotel Type")
            ax.set_ylabel("Cancellation Rate (%)")
            ax.tick_params(axis="x", rotation=0)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Deep dive
    st.header("2. Booking & Revenue Deep Dive")
    col1, col2 = st.columns(2)

    with col1:
        if "market_segment" in filtered.columns:
            seg = filtered["market_segment"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            seg.sort_values().plot(kind="barh", ax=ax)
            ax.set_title("Bookings by Market Segment")
            ax.set_xlabel("Bookings")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        if "arrival_date_month" in filtered.columns:
            month_cancel = (
                filtered.groupby(["arrival_month_num", "arrival_date_month"])["is_canceled"]
                .mean().mul(100).reset_index()
                .sort_values("arrival_month_num")
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(month_cancel["arrival_date_month"], month_cancel["is_canceled"], marker="o")
            ax.set_title("Cancellation Rate by Arrival Month")
            ax.set_ylabel("Cancellation Rate (%)")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if {"deposit_type", "is_canceled"}.issubset(filtered.columns):
        deposit = filtered.groupby("deposit_type")["is_canceled"].mean().mul(100).sort_values(ascending=False)
        st.subheader("Cancellation Rate by Deposit Type")
        st.dataframe(
            deposit.rename("Cancellation Rate (%)").round(2).to_frame(),
            use_container_width=True
        )

    # Prediction
    st.header("3. Cancellation Prediction")
    st.write(
        "Two classification models are trained using booking-time information. "
        "Outcome-revealing reservation-status fields are excluded to reduce target leakage."
    )

    results, comparison, best_name, X_test, y_test = train_models(df)

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True
    )

    st.success(f"Selected model for the risk demo: {best_name}")

    best = results[best_name]
    cm = best["cm"]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm)
        ax.set_title(f"{best_name} — Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["Not Canceled", "Canceled"])
        ax.set_yticks([0, 1], ["Not Canceled", "Canceled"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Model Metrics")
        metrics = {
            "Accuracy": best["accuracy"],
            "Precision": best["precision"],
            "Recall": best["recall"],
            "F1 Score": best["f1"],
            "ROC-AUC": best["roc_auc"]
        }
        for k, v in metrics.items():
            st.metric(k, f"{v:.3f}")

    # Single-booking risk demo
    st.subheader("Single Booking Risk Demo")
    feature_defaults = {
        "hotel": "City Hotel",
        "lead_time": 100,
        "arrival_date_year": int(df["arrival_date_year"].median()),
        "arrival_date_month": "July",
        "arrival_date_week_number": int(df["arrival_date_week_number"].median()),
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 0.0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": float(df["adr"].median()),
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }

    cols = [c for c in feature_defaults if c in X_test.columns]
    left, right = st.columns(2)
    input_data = {}

    with left:
        for c in cols[:len(cols)//2]:
            if X_test[c].dtype == "object":
                options = sorted(df[c].dropna().astype(str).unique().tolist())
                default = str(feature_defaults[c]) if str(feature_defaults[c]) in options else options[0]
                input_data[c] = st.selectbox(c.replace("_", " ").title(), options, index=options.index(default))
            else:
                input_data[c] = st.number_input(
                    c.replace("_", " ").title(),
                    value=float(feature_defaults[c]),
                    step=1.0
                )

    with right:
        for c in cols[len(cols)//2:]:
            if X_test[c].dtype == "object":
                options = sorted(df[c].dropna().astype(str).unique().tolist())
                default = str(feature_defaults[c]) if str(feature_defaults[c]) in options else options[0]
                input_data[c] = st.selectbox(c.replace("_", " ").title(), options, index=options.index(default))
            else:
                input_data[c] = st.number_input(
                    c.replace("_", " ").title(),
                    value=float(feature_defaults[c]),
                    step=1.0
                )

    if st.button("Predict Cancellation Risk", type="primary"):
        sample = pd.DataFrame([input_data])
        probability = best["model"].predict_proba(sample)[0, 1]
        prediction = int(probability >= 0.5)

        if probability >= 0.60:
            risk = "HIGH"
        elif probability >= 0.30:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        st.metric("Predicted Cancellation Probability", f"{probability:.1%}")
        st.metric("Risk Segment", risk)
        st.write(
            f"Model classification: **{'Likely Canceled' if prediction else 'Likely Not Canceled'}**"
        )

        if risk == "HIGH":
            st.warning(
                "Business action: consider proactive confirmation and closer monitoring of this reservation."
            )
        elif risk == "MEDIUM":
            st.info(
                "Business action: consider a reminder/confirmation workflow and monitor the booking."
            )
        else:
            st.success(
                "Business action: normal reservation handling is appropriate based on this model output."
            )

    # Business insights
    st.header("4. Business Insights & Actions")

    if "lead_time" in filtered.columns:
        q = filtered["lead_time"].quantile([0.33, 0.67]).tolist()
        bins = [-np.inf, q[0], q[1], np.inf]
        labels = ["Short Lead Time", "Medium Lead Time", "Long Lead Time"]
        temp = filtered.copy()
        temp["lead_time_group"] = pd.cut(temp["lead_time"], bins=bins, labels=labels)
        lt = temp.groupby("lead_time_group", observed=False)["is_canceled"].mean().mul(100)

        st.subheader("Fact → Insight → Action")
        highest_group = lt.idxmax()
        highest_rate = lt.max()

        st.markdown(
            f"""
**Fact:** The observed cancellation rate differs across lead-time groups; the highest
observed group in the filtered data is **{highest_group} ({highest_rate:.1f}%)**.

**Insight:** Lead time is associated with different observed cancellation rates in this dataset.

**Risk/Opportunity:** Booking segments with higher observed cancellation rates may create
greater uncertainty for occupancy planning.

**Action:** The hotel could use predicted cancellation risk to prioritize confirmation
messages and review inventory planning for higher-risk reservations.
"""
        )

    st.caption(
        "Note: risk thresholds in the demo are analytical segments for this project; "
        "they are not official hotel policy or causal conclusions."
    )


if __name__ == "__main__":
    main()
