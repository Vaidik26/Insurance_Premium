# 🛡️ Insurance Premium Prediction

This project predicts the **insurance premium amount** a customer should be charged based on features like age, sex, BMI, smoking status, region, and number of children. The goal is to use machine learning to build a regression model that helps insurance companies set accurate premiums.

## 📌 Problem Statement

Insurance companies need to calculate the premium amount for new customers based on certain features. Manual calculation can be inconsistent and inaccurate. This project automates premium prediction using supervised machine learning techniques.

---

## 🧠 ML Task

- **Type**: Supervised Regression
- **Model**: CatBoost Regressor (selected for speed and generalization)
- **Target Column**: `expenses` (Insurance Premium)
- **Input Features**: `age`, `sex`, `bmi`, `children`, `smoker`, `region`

---

## 📁 Project Structure

Insurance_Premium_Prediction/
│
├── artifacts/ # Contains trained model, preprocessor, metrics, etc.
├── data/ # Raw and processed data
├── notebooks/ # Jupyter notebooks for EDA & experimentation
├── src/
│ └── Insurance/ # Core source code package
│ ├── components/ # Data ingestion, transformation, training modules
│ ├── utils.py # Utility functions
│ ├── logger.py # Custom logging
│ └── exception.py # Custom exceptions
│
├── app.py # Flask application for prediction
├── requirements.txt # Python dependencies
├── setup.py # Setup script for packaging
├── README.md # Project documentation
└── .gitignore # Files ignored by Git


## 🖥️ User Interface (UI)


![Screenshot 2025-06-20 112853](https://github.com/user-attachments/assets/a4bc566b-740c-4282-866a-9dfcb8a52d14)


![Screenshot 2025-06-20 112908](https://github.com/user-attachments/assets/64f78fef-df87-4551-b915-0e4e566c5cf4)



## ⚙️ Setup Instructions

1. Clone the Repository

git clone https://github.com/your-username/Insurance_Premium_Prediction.git

cd Insurance_Premium_Prediction

2. Create and Activate Virtual Environment

Create virtual environment
python -m venv venv

Activate on Windows
venv\Scripts\activate

Activate on Unix or Mac
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Train the Model
python src/Insurance/pipeline/training.py

#This will:

1. Ingest the data

2. Transform features (encoding, scaling)

3. Train the CatBoost model using RandomizedSearchCV

4. Save model and metrics to artifacts/


## 🚀 Run the Web App (Flask)

python app.py

Navigate to http://127.0.0.1:5000 in your browser.

Enter details in the form

See results and download predictions

Logging and error handling are included


## ✅ Features

🧼 Robust preprocessing (missing values, encoding, scaling)

🚀 Fast and generalizable model with CatBoost + RandomizedSearchCV

📊 Cross-validation and metric tracking (MAE, RMSE, R²)

🌐 Web interface using Flask

🐞 Custom logging and exception handling

📁 Modular and reusable code

📉 Sample Input Data

age	sex	bmi	children	smoker	region
45	male	29.8	2	yes	southeast


## Prediction Output:

Predicted Premium: ₹27,845.12


## 📈 Evaluation Metrics

Stored in artifacts/metrics.json after training.

MAE: 2631.18

RMSE: 4152.67

R² Score: 0.87


## 📌 Future Work

Add DVC for pipeline versioning

CI/CD with GitHub Actions

Containerization with Docker

Deploy to AWS/GCP


## 🙌 Acknowledgments

Kaggle Dataset

CatBoost Documentation

Flask for web deployment


## 📬 Contact

Vaidik Yadav

📧 vaidiky90@gmail.com

🌐 LinkedIn : https://www.linkedin.com/in/vaidik-yadav-260a60248/

## 📝 License

This project is licensed under the MIT License.
