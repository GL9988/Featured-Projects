# 💰 Insurance Premium Prediction  
Predict client medical risk and insurance costs using supervised machine learning models.

---

## 🚀 Motivation  
Built to support smarter premium pricing through risk stratification. This project empowers insurers to dynamically forecast costs and optimize profitability while ensuring fair and transparent customer segmentation.

---

## 📦 Dataset  
- **Source**: [Kaggle – Insurance Premium Dataset](https://www.kaggle.com/datasets/simranjain17/insurance/data)  
- **Records**: 1,300+ client entries  
- **Features**: Age, Gender, BMI, Children, Smoker, Region, Charges

---

## 🧠 Model Architectures  
Applied and compared 4 models to predict insurance charges:

| Model              | Purpose                       | Notes                                         |
|-------------------|-------------------------------|-----------------------------------------------|
| Linear Regression | Baseline model                | Interpretable coefficients                    |
| Lasso Regression  | Feature selection             | Reduced multicollinearity                     |
| Random Forest     | Nonlinear ensemble prediction | ⭐ Best performer (R² ≈ 0.83)                  |
| KNN               | Local relationship modeling   | Flexible but less interpretable               |

---

## 📊 Key Insights  
- **Top Drivers of Charges**: `smoker status`, `age`, `BMI`  
- **Bias Caution**: Gender and region can skew results—fairness addressed with feature scaling and dummy encoding  
- **Random Forest selected for deployment** due to high predictive accuracy and low RMSE
<img width="244" alt="image" src="https://github.com/user-attachments/assets/0041b517-3f02-4447-b65f-bd0e08463a99" />

---

## 📈 Model Evaluation  

| Model              | R² Score | RMSE    |
|-------------------|----------|---------|
| Linear Regression | 0.7270   | 0.5117  |
| Lasso Regression  | 0.7266   | 0.5121  |
| Random Forest     | 0.8292   | 0.4046  |
| KNN               | 0.5230   | 0.6763  |

> ✅ **Best Model**: Random Forest  
> ✅ **Business Value**: More accurate risk profiling leads to better pricing strategies and higher ROI.

---

## 🛠 Tech Stack  
- **Languages**: Python, R  
- **Libraries**: `scikit-learn`, `NumPy`, `pandas`, `matplotlib`, `seaborn`  
- **Tools**: Excel, SPSS, Kaggle, Jupyter Notebook

---

## 💼 Business Impact  
- Enabled segmentation of high-risk vs. low-risk clients  
- Supported dynamic premium pricing aligned with real-world behavior  
- Highlighted cost-saving opportunities by modeling potential churn and optimizing lifetime value  

---

## ▶️ Run This Project  
```bash
git clone https://github.com/YourUsername/Insurance-Premium-Prediction
cd insurance-premium
pip install -r requirements.txt
python model_train.py
