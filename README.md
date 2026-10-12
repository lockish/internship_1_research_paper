# Predictive Modeling and Risk Scoring for Bank Customer Churn

*A machine-learning research project with explainable risk scores and what-if analysis*

**Unified Mentor Internship Project** | Prepared by Lokesh

---

## Overview

Customer churn erodes revenue and customer lifetime value in retail banking, yet most retention effort starts only after a customer has already decided to leave. This project builds a predictive churn-intelligence workflow that gives every customer a churn probability, so retention teams can act earlier and focus on the customers most likely to leave.

The full write-up is in the research paper included in this repository.

## Dataset

- 10,000 retail bank customers across France, Germany and Spain
- 14 original columns; the target variable is `Exited` (1 = churned, 0 = retained)
- 20.4% of customers churned (2,037 of 10,000), so the classes are imbalanced
- No missing values and no duplicate customer rows after removing identifier columns

Features include CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember and EstimatedSalary.

## Objectives

**Primary**
- Predict customer churn with high accuracy
- Generate churn probability scores
- Identify the key churn drivers

**Secondary**
- Reduce false positives in churn detection
- Improve the interpretability of machine-learning models
- Enable scenario-based churn risk analysis

## Methodology

1. **Preprocessing:** removed non-informative columns (CustomerId, Surname, constant Year), one-hot encoded Geography and Gender, and standardised numeric features for logistic regression.
2. **Feature engineering:** BalanceSalaryRatio, ProductDensity, EngagementProduct, AgeTenure and TenureAgeRatio.
3. **Train-test strategy:** 80/20 stratified split on `Exited`, with five-fold stratified cross-validation on the training set for model selection and threshold tuning.
4. **Models compared:** Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.
5. **Metrics:** accuracy, precision, recall, F1-score and ROC-AUC.
6. **Explainability:** permutation importance and partial dependence plots.
7. **Risk scoring:** predicted probabilities grouped into Low (below 25%), Medium (25-50%) and High (50% or above) segments.

## Results

### Model comparison (test set, threshold 0.50)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.813 | 0.624 | 0.204 | 0.307 | 0.777 |
| Decision Tree | 0.861 | 0.788 | 0.430 | 0.556 | 0.847 |
| Random Forest | 0.868 | 0.812 | 0.457 | 0.585 | 0.863 |
| **Gradient Boosting** | **0.868** | **0.777** | **0.489** | **0.600** | **0.869** |

Gradient Boosting was selected because it had the highest cross-validated ROC-AUC (0.862).

### Tuned decision threshold

| Operating point | Threshold | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Default | 0.50 | 0.868 | 0.777 | 0.489 | 0.600 |
| Tuned (max F1) | 0.37 | 0.863 | 0.688 | 0.595 | 0.638 |

### Risk segments (test set)

| Segment | Share of customers | Observed churn rate | Share of all churners |
| --- | --- | --- | --- |
| Low (<25%) | 74.4% | 8.0% | 29.2% |
| Medium (25-50%) | 12.8% | 34.8% | 21.9% |
| High (>=50%) | 12.8% | 77.7% | 48.9% |

The high-risk segment holds 12.8% of customers but 48.9% of all churners.

## Key Findings

- **Top churn drivers:** Age, NumOfProducts, Geography (Germany), Balance and EngagementProduct.
- **By country:** Germany has the highest churn (32.4%) and France the lowest (16.2%).
- **By activity:** inactive members churn at 26.9% versus 14.3% for active members.
- **By age:** the 51-60 age band churns most (56.2%); the 18-30 band least (7.5%).
- **By products:** churn rises sharply for customers holding 3 or 4 products.

## Recommendations

- Focus retention offers on the high-risk segment instead of contacting everyone.
- Run re-engagement programmes for inactive members.
- Review the product journey for customers holding an unusual number of products.
- Investigate market-specific attrition in Germany.
- Build life-stage programmes for the 51-60 age group.
- Check fairness, monitor score drift and retrain the model regularly.

## Limitations and Future Work

- The data is a single snapshot without timestamps, so behaviour over time cannot be modelled.
- Relationships are correlational; retention actions should be validated with A/B tests.
- Probabilities are not explicitly calibrated.
- Cost-sensitive thresholds were not modelled.
- Future work: XGBoost/LightGBM, SHAP-based explanations, survival analysis and uplift modelling.

## Repository Contents

```
.
├── Bank_Churn_Research_Paper.pdf   # Full research paper
└── README.md                       # Project overview
```

## Tools and Libraries

Python, scikit-learn, pandas, NumPy and Matplotlib. The paper also specifies a Streamlit risk-scoring dashboard with a churn risk calculator, probability distribution view, feature importance charts and a what-if scenario simulator.

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
