# HOTEL BOOKING INSIGHTS

### Data Analytics | Machine Learning | Business Intelligence

> An end-to-end analytics project that transforms hotel booking data into cancellation-risk insights and business actions.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## PROJECT SNAPSHOT

| | |
|---|---|
| **Domain** | Hospitality Analytics |
| **Dataset** | Hotel Booking Demand |
| **Records** | 119,390 |
| **Prediction Target** | Booking Cancellation |
| **Models** | Logistic Regression, Random Forest |
| **Application** | Interactive Streamlit Dashboard |
| **Primary Output** | Cancellation Risk Prediction |

---

## THE BUSINESS QUESTION

Hotels lose revenue and occupancy predictability when reservations are cancelled.

This project asks:

> **Can historical booking patterns be used to identify reservations with higher cancellation risk — and turn that prediction into a useful business action?**

The project goes beyond descriptive charts by connecting analytics with predictive modelling and decision-making.

```text
RAW DATA
    |
    v
DATA CLEANING
    |
    v
FEATURE ENGINEERING
    |
    v
EXPLORATORY ANALYSIS
    |
    v
KPI ANALYSIS
    |
    v
CANCELLATION PREDICTION
    |
    v
RISK SEGMENTATION
    |
    v
BUSINESS ACTION
```

---

# KEY RESULTS

<table>
<tr>
<td align="center" width="20%">

### 78.0%

Accuracy

</td>
<td align="center" width="20%">

### 88.5%

ROC-AUC

</td>
<td align="center" width="20%">

### 83.7%

Recall

</td>
<td align="center" width="20%">

### 67.7%

F1 Score

</td>
<td align="center" width="20%">

### 119K+

Bookings

</td>
</tr>
</table>

> **Why these metrics matter:** Cancellation detection benefits from identifying as many genuinely cancellable bookings as possible, while maintaining a reasonable balance between precision and recall.

---

# WHAT THE PROJECT DELIVERS

### 01 — Executive Overview

A high-level view of hotel booking activity.

**Includes:**

- Total bookings
- Cancellation rate
- Average Daily Rate
- Average stay duration
- Estimated booking value
- Monthly booking demand

---

### 02 — Booking & Revenue Deep Dive

Explores patterns behind hotel bookings and cancellations.

**Analyzes:**

- Market segments
- Hotel type
- Arrival months
- Lead time
- Deposit type
- Booking behaviour
- Cancellation patterns

---

### 03 — Cancellation Prediction

Two machine-learning models are trained and compared:

**Logistic Regression**

Used as the baseline classification model.

**Random Forest**

Used to capture more complex relationships between booking characteristics.

The application also provides an interactive **single-booking risk demonstration**.

---

### 04 — Business Insights & Actions

The final layer converts analytical findings into business-oriented recommendations.

The reasoning framework is:

```text
FACT
  ↓
INSIGHT
  ↓
RISK / OPPORTUNITY
  ↓
ACTION
```

This keeps the project focused on **what the data means for a business**, rather than only presenting charts.

---

# MACHINE LEARNING

## Target Variable

```text
is_canceled
```

The model predicts whether a hotel reservation will be cancelled.

### Models Compared

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Non-linear classification model |

### Random Forest Results

| Metric | Result |
|---|---:|
| Accuracy | **78.0%** |
| Precision | **56.8%** |
| Recall | **83.7%** |
| F1 Score | **67.7%** |
| ROC-AUC | **88.5%** |

---

# MODEL EVALUATION

The application includes a confusion matrix to show how the model performs across both classes.

| | Predicted: Not Cancelled | Predicted: Cancelled |
|---|---:|---:|
| **Actual: Not Cancelled** | 9,614 | 3,061 |
| **Actual: Cancelled** | 782 | 4,023 |

The model's evaluation is presented directly in the dashboard alongside the classification metrics.

---

# PREDICTIVE RISK DEMONSTRATION

The dashboard allows a user to enter booking characteristics and obtain an estimated cancellation probability.

Example project run:

```text
Predicted Cancellation Probability
              77.2%

Risk Segment
              HIGH

Classification
              Likely Cancelled
```

The application then translates the prediction into a potential business response:

> **Consider proactive confirmation and closer monitoring of the reservation.**

This demonstrates how a machine-learning output can be connected to an operational decision.

---

# BUSINESS INSIGHT EXAMPLE

### FACT

The highest observed cancellation rate among the lead-time groups in the current dashboard analysis is **32.7% for Long Lead Time bookings**.

### INSIGHT

Lead time is associated with different observed cancellation rates in the dataset.

### RISK / OPPORTUNITY

Booking segments with higher observed cancellation rates may introduce greater uncertainty into occupancy planning.

### ACTION

Hotels could use predicted cancellation risk to prioritize confirmation messages and review inventory planning for higher-risk reservations.

> **Note:** These are analytical associations observed in the dataset and are not presented as proof of causation.

---

# DATA PREPARATION

The project applies a complete data-preparation workflow:

```text
Data Loading
     |
     v
Quality Checks
     |
     v
Missing-Value Treatment
     |
     v
Duplicate Removal
     |
     v
Feature Engineering
     |
     v
Categorical Encoding
     |
     v
Train / Test Split
```

### Leakage Prevention

The prediction target is:

```text
is_canceled
```

Variables that directly reveal the eventual booking outcome are deliberately excluded from the predictive feature set.

Specifically:

```text
reservation_status
reservation_status_date
```

This reduces the risk of target leakage and keeps the prediction task aligned with information that would be available before the final booking outcome.

---

# DASHBOARD

The project is implemented as an interactive **Streamlit** application.

The dashboard provides:

- KPI cards
- Interactive hotel-type filtering
- Booking-demand visualizations
- Cancellation-rate analysis
- Market-segment analysis
- Model comparison
- Confusion matrix
- Individual booking-risk prediction
- Business recommendations

The interface is designed around a simple principle:

> **A dashboard should communicate a decision, not just display data.**

---

# TECH STACK

### Data Analytics

`Python` `Pandas` `NumPy`

### Data Visualization

`Matplotlib`

### Machine Learning

`Scikit-learn`

- Logistic Regression
- Random Forest
- Train/Test Split
- Classification Metrics
- ROC-AUC

### Application

`Streamlit`

### Development

`VS Code` `Git` `GitHub`

---

# PROJECT STRUCTURE

```text
Hotel_Booking_Insights/
│
├── hotel_booking_insights.py
│
├── hotel_bookings.csv
│
├── requirements.txt
│
├── Project_Report.docx
│
├── README.md
│
└── .gitignore
```

| File | Description |
|---|---|
| `hotel_booking_insights.py` | Complete analytics, visualization, ML and Streamlit application |
| `hotel_bookings.csv` | Raw dataset used for the project |
| `requirements.txt` | Required Python dependencies |
| `Project_Report.docx` | Detailed project documentation |
| `README.md` | Project overview and technical documentation |
| `.gitignore` | Files excluded from version control |

---

# DATASET

### Hotel Booking Demand Dataset

**Source:** Kaggle

https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

The dataset contains hotel reservation records covering areas such as:

- Booking timing
- Arrival information
- Length of stay
- Customer characteristics
- Market segment
- Deposit type
- Pricing
- Previous booking behaviour
- Cancellation status

---

# RUN LOCALLY

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Hotel_Booking_Insights
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the dashboard

```bash
streamlit run hotel_booking_insights.py
```

The application will open locally at:

```text
http://localhost:8501
```

---

# REQUIREMENTS

The project uses:

```text
pandas
numpy
matplotlib
scikit-learn
streamlit
```

All required packages are listed in:

```text
requirements.txt
```

---

# PROJECT CONTEXT

This project was developed as part of the:

**IBM SkillsBuild Data Analytics with AI Internship**

The implementation follows the analytical workflow covered during the internship:

```text
Raw Data
   ↓
Cleaning
   ↓
EDA
   ↓
Prediction
   ↓
Business Insights
   ↓
Action
```

The goal is to demonstrate the complete journey from a raw dataset to a business-oriented analytical solution.

---

# FUTURE IMPROVEMENTS

Potential next steps include:

- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- SHAP-based model explainability
- Revenue-loss estimation from predicted cancellations
- Customer segmentation
- Advanced time-series analysis
- Cloud deployment
- Automated cancellation-risk monitoring

---

# AUTHOR

## Drishita Paul

**B.Tech — Electronics & Computer Science**

**Focus Areas**

`Software Engineering` · `Cloud` · `Data Analytics` · `AI/ML`

---

## PROJECT PHILOSOPHY

```text
DATA
  ↓
UNDERSTAND
  ↓
PREDICT
  ↓
DECIDE
  ↓
ACT
```

**The objective is not simply to build a model.**

**The objective is to turn data into a decision.**