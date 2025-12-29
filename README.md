# 🚗💰 USED CAR PRICE PREDICTOR 💰🚗

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=F75C7E&center=true&vCenter=true&width=800&lines=Predict+Used+Car+Prices+with+Machine+Learning;Random+Forest+Regression+on+CarDekho+Dataset;45%2C000%2B+Car+Listings+Analyzed;Interactive+Streamlit+Dashboard+%F0%9F%94%A5)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F79310E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[![Live Demo](https://img.shields.io/badge/🎯_Live_Streamlit_App-Click_Here-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://used-car-price-modeling-project.streamlit.app/)
[![Kaggle Notebook](https://img.shields.io/badge/📊_Kaggle_Notebook-View_Analysis-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mayankdataanalyst/car-price-predictor-notebook)
[![Dataset](https://img.shields.io/badge/📂_CarDekho_Dataset-45K+_Listings-1DA1F2?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho/code)

![Car Animation](https://user-images.githubusercontent.com/74038190/212748830-4c709398-a386-4761-84d7-9e10b98fbe6e.gif)

### 🚗 **Predict the perfect price for any used car** using **Random Forest Regression + Feature Engineering** 🤖

### 💰 45,000+ CarDekho Listings × AI = **Smart Pricing Intelligence** 💡

---

## 🌟 **WHAT IS THIS?** 🌟

<table>
<tr>
<td width="50%">

### 🎯 **The Magic**

This **ML-powered car price predictor** estimates fair market value of used cars using **Random Forest Regressor** trained on 45,000+ CarDekho listings. Input vehicle details (year, fuel type, transmission, mileage, etc.) and get instant price predictions with interactive visualizations!

**Think of it as:**
- 🧠 Brain = Random Forest Regressor
- 📊 Input = Car Features (Year, Mileage, Brand, Fuel Type)  
- 💰 Output = Predicted Selling Price (₹)

</td>
<td width="50%">

### 🔥 **Key Features**

✅ Random Forest with hyperparameter tuning  
✅ Handles categorical & numerical features  
✅ Feature importance analysis  
✅ Interactive Plotly visualizations  
✅ **Real-time price predictions** 🕒  
✅ Beautiful Streamlit UI with mobile support  

**Performance Metrics:**
- 📉 **MAE**: ₹50,000-70,000  
- 📊 **RMSE**: ₹80,000-100,000  
- 🎯 **R²**: 0.92+ (High accuracy)

</td>
</tr>
</table>

---

## 🛠️ **TECH STACK** 🛠️

![Tech Stack](https://skillicons.dev/icons?i=python,github,vscode,git)

| **Category** | **Technologies** |
|-------------|-----------------|
| 🐍 **Language** | Python 3.8+ |
| 📊 **Data Science** | Pandas, NumPy, Scikit-learn |
| 🎨 **Frontend** | Streamlit |
| 📈 **Visualization** | Plotly, Matplotlib, Seaborn |
| 🧪 **Model** | Random Forest Regressor, GridSearchCV |
| 🔧 **Preprocessing** | LabelEncoder, StandardScaler, OneHotEncoder |
| 💾 **Serialization** | Joblib, Pickle |
| 📦 **Data Source** | CarDekho (Kaggle) - 45,000+ listings |

---

## 📂 **PROJECT STRUCTURE** 📂

```
🚗 used-car-price-modeling/
│
├── 📁 app.py                         # Streamlit web application (19KB)
├── 📦 requirements.txt               # Python dependencies
├── 📖 README.md                      # You are here!
│
├── 🔗 External Resources:
│   ├── 🎯 Streamlit App              # Live deployment
│   ├── 📊 Kaggle Notebook            # Model training & EDA
│   └── 📂 CarDekho Dataset           # 45,000+ car listings
```

**Note:** This repository contains the **production deployment code**. Full training pipeline, EDA, and hyperparameter tuning are available in the [Kaggle Notebook](https://www.kaggle.com/code/mayankdataanalyst/car-price-predictor-notebook).

---

## 🚀 **QUICK START** 🚀

![Rocket](https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-705f7be0b224.gif)

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/mayank-goyal09/used-car-price-modeling.git
cd used-car-price-modeling
```

### **Step 2: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 3: Run the App** 🎯

```bash
streamlit run app.py
```

### **Step 4: Open in Browser** 🌐

The app will automatically open at: **`http://localhost:8501`**

---

## 🎮 **HOW TO USE** 🎮

<table>
<tr>
<td width="50%">

### 🔹 **Quick Prediction Mode**

1. Open the [Streamlit App](https://used-car-price-modeling-project.streamlit.app/)
2. Input car details:
   - **Year** (Manufacturing year)
   - **Present Price** (₹ Lakh - current showroom price)
   - **Kms Driven** (Total mileage)
   - **Fuel Type** (Petrol/Diesel/CNG)
   - **Seller Type** (Dealer/Individual)
   - **Transmission** (Manual/Automatic)
   - **Owner** (First/Second/Third owner)
3. Click **"Predict Price"** 💰
4. View estimated selling price with confidence intervals!

</td>
<td width="50%">

### 🔹 **Insights Dashboard** 📊

1. Navigate to **"Model Insights"** tab
2. Explore feature importance:
   - Which features most affect price?
   - Visual breakdown with bar charts
3. Analyze prediction confidence
4. Compare actual vs. predicted prices (scatter plot)

</td>
</tr>
</table>

---

## 🧪 **HOW IT WORKS** 🧪

```mermaid
graph LR
    A[CarDekho Dataset] --> B[Data Cleaning & EDA]
    B --> C[Feature Engineering]
    C --> D[Train-Test Split]
    D --> E[Random Forest Training]
    E --> F[Hyperparameter Tuning]
    F --> G[Model Evaluation]
    G --> H[Streamlit Deployment]
    H --> I[Real-Time Predictions]
```

### **Pipeline Breakdown:**

1️⃣ **Data Collection** → 45,000+ used car listings from CarDekho  
2️⃣ **Feature Engineering** → Encode categorical features:
   - Fuel_Type (Petrol/Diesel/CNG)
   - Seller_Type (Dealer/Individual)
   - Transmission (Manual/Automatic)
   - One-hot encoding for brands  
3️⃣ **Model Training** → Random Forest Regressor with GridSearchCV  
4️⃣ **Hyperparameter Optimization** → Best n_estimators, max_depth, min_samples_split  
5️⃣ **Evaluation** → MAE, RMSE, R² on test set (80/20 split)  
6️⃣ **Deployment** → Streamlit app with Plotly visualizations  

---

## 📊 **DATASET & FEATURES** 📊

![Data Analysis](https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif)

### **Dataset Overview**

- 📍 **Source**: [CarDekho (Kaggle)](https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho/code)
- 📏 **Size**: 45,000+ used car listings
- 🌍 **Region**: India (All major cities)
- 🎯 **Target Variable**: `Selling_Price` (₹ in Lakhs)

### **Feature Categories**

| **Feature** | **Type** | **Description** |
|------------|---------|---------------|
| 🗓️ **Year** | Numerical | Manufacturing year (2003-2020) |
| 💰 **Present_Price** | Numerical | Current showroom price (₹ Lakh) |
| 🚙 **Kms_Driven** | Numerical | Total kilometers driven |
| ⛽ **Fuel_Type** | Categorical | Petrol, Diesel, CNG |
| 👤 **Seller_Type** | Categorical | Dealer, Individual |
| ⚙️ **Transmission** | Categorical | Manual, Automatic |
| 🔑 **Owner** | Categorical | First/Second/Third owner |

### **Top 3 Predictive Features** (from Feature Importance)

1. 💰 **Present_Price** → Strongest predictor (60%+ importance)
2. 🗓️ **Year** → Vehicle age matters (20%+ importance)
3. 🚙 **Kms_Driven** → Mileage impact (10%+ importance)

---

## 🎨 **FEATURES SHOWCASE** 🎨

### ✨ **What Makes This Special?**

```python
# Feature Highlights

features = {
    "Interactive Predictions": "💰 Instant price estimation",
    "Feature Importance": "📊 Visual breakdown of key factors",
    "Mobile Friendly": "📱 Responsive UI with clean layout",
    "No Complex Inputs": "✅ Simple dropdown selectors",
    "Premium Charts": "🎨 Professional Plotly visualizations",
    "Real-Time Updates": "🔄 Instant recalculation",
    "Confidence Intervals": "📈 Prediction uncertainty ranges",
}
```

### **App Sections:**

1. **💰 Price Predictor** → Fast input with dropdowns
2. **📈 Model Insights** → Feature importance dashboard
3. **🚗 Car Details** → Comprehensive input form
4. **🧠 Prediction Confidence** → Uncertainty visualization

---

## 💡 **BUSINESS USE CASES** 💡

![Business Use Cases](https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif)

### **Who Benefits from This?**

- 🚗 **Car Buyers**: Avoid overpaying - get fair market estimates
- 💼 **Dealers**: Price inventory competitively
- 📊 **Market Analysts**: Track depreciation trends
- 💰 **Loan Officers**: Assess vehicle collateral value
- 🔧 **Insurance Companies**: Determine replacement costs
- 📱 **E-Commerce Platforms**: Auto-price listings (OLX, CarDekho)

---

## 📈 **MODEL PERFORMANCE** 📈

### **Evaluation Metrics:**

| **Metric** | **Value** | **Interpretation** |
|-----------|---------|-------------------|
| **MAE** | ₹50,000-70,000 | Average error of ₹50K-70K |
| **RMSE** | ₹80,000-100,000 | Low error for price prediction |
| **R² Score** | 0.92+ | Excellent predictive power |
| **CV Score** | Consistent | Robust across test folds |

### **Sample Predictions:**

| **Car Details** | **Actual Price** | **Predicted Price** | **Error** |
|----------------|-----------------|-------------------|----------|
| 2016 Hyundai i20 Sportz (Manual, Petrol) | ₹5.5 Lakh | ₹5.3 Lakh | -₹0.2 Lakh |
| 2018 Maruti Swift VXI (Manual, Petrol) | ₹6.0 Lakh | ₹6.2 Lakh | +₹0.2 Lakh |
| 2015 Honda City (Automatic, Diesel) | ₹7.2 Lakh | ₹7.0 Lakh | -₹0.2 Lakh |

*Sample data - actual results vary by specific vehicle features*

---

## 📚 **SKILLS DEMONSTRATED** 📚

- ✅ **Supervised Learning**: Random Forest Regression
- ✅ **Feature Engineering**: Encoding categorical variables
- ✅ **Hyperparameter Tuning**: GridSearchCV optimization
- ✅ **Model Evaluation**: MAE, RMSE, R², cross-validation
- ✅ **Data Visualization**: Plotly interactive charts
- ✅ **Web Development**: Streamlit app with custom CSS
- ✅ **Python**: Pandas, NumPy, Scikit-learn
- ✅ **Data Preprocessing**: Scaling, encoding, handling missing values
- ✅ **Deployment**: Production-ready web app

---

## 🔮 **FUTURE ENHANCEMENTS** 🔮

- [ ] Add image-based condition assessment (computer vision)
- [ ] Implement XGBoost/LightGBM for comparison
- [ ] Add SHAP values for explainable AI
- [ ] Create REST API endpoint for integration
- [ ] Add price trend forecasting (time-series)
- [ ] Implement recommendation system (similar cars)
- [ ] Build mobile app version (React Native)
- [ ] Add real-time market data scraping

---

## 🤝 **CONTRIBUTING** 🤝

![Contributing](https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif)

Contributions are **always welcome**! 🎉

1. 🍴 Fork the Project
2. 🌱 Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the Branch (`git push origin feature/AmazingFeature`)
5. 🎁 Open a Pull Request

---

## 📝 **LICENSE** 📝

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 **CONNECT WITH ME** 👨‍💻

[![GitHub](https://img.shields.io/badge/GitHub-mayank--goyal09-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank%20Goyal-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
[![Kaggle](https://img.shields.io/badge/Kaggle-mayankdataanalyst-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/mayankdataanalyst)
[![Email](https://img.shields.io/badge/Email-itsmaygal09%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:itsmaygal09@gmail.com)

**Mayank Goyal**  
📊 Data Analyst | 🤖 ML Enthusiast | 🐍 Python Developer  
💼 Data Analyst Intern @ SpacECE Foundation India

---

## ⭐ **SHOW YOUR SUPPORT** ⭐

![Support](https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif)

Give a ⭐️ if this project helped you understand car price prediction with ML!

### 🚗 **Built with Data & ❤️ by Mayank Goyal** 🚗

**"Turning car listings into smart pricing intelligence, one prediction at a time!"** 📊

---

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)
