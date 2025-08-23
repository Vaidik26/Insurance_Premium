# 🛡️ Insurance Premium Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-CatBoost-orange.svg)
![Web App](https://img.shields.io/badge/Web%20App-Flask-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*A sophisticated machine learning solution for accurate insurance premium prediction*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Technical Architecture](#-technical-architecture)
- [Features](#-features)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [Contact](#-contact)

---

## 🎯 Overview

This project leverages **advanced machine learning techniques** to predict insurance premium amounts with high accuracy. By analyzing key customer attributes such as age, sex, BMI, smoking status, region, and number of children, our system provides insurance companies with data-driven premium recommendations.

### 🎯 Key Objectives
- **Automate** premium calculation processes
- **Improve** accuracy and consistency in pricing
- **Reduce** manual calculation errors
- **Enhance** customer experience with transparent pricing

---

## 🔍 Problem Statement

Insurance companies face significant challenges in manually calculating premium amounts for new customers. This approach often leads to:
- **Inconsistent pricing** across similar risk profiles
- **Human error** in calculations
- **Time-consuming** manual processes
- **Lack of standardization** in premium determination

Our solution addresses these challenges through intelligent automation and machine learning-driven insights.

---

## 🏗️ Technical Architecture

### 🤖 Machine Learning Framework
- **Task Type**: Supervised Regression
- **Primary Model**: CatBoost Regressor
- **Optimization**: RandomizedSearchCV for hyperparameter tuning
- **Target Variable**: `expenses` (Insurance Premium Amount)

### 🔧 Core Technologies
- **Backend**: Python 3.8+, Flask
- **ML Libraries**: CatBoost, Scikit-learn, Pandas, NumPy
- **Data Processing**: Custom preprocessing pipeline
- **Deployment**: Web-based interface with RESTful API

### 📊 Feature Engineering
| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Customer's age in years |
| `sex` | Categorical | Gender (male/female) |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of dependents |
| `smoker` | Categorical | Smoking status (yes/no) |
| `region` | Categorical | Geographic region |

---

## ✨ Features

### 🚀 **Core Capabilities**
- **Intelligent Preprocessing**: Robust handling of missing values, categorical encoding, and feature scaling
- **Advanced Modeling**: CatBoost algorithm with cross-validation and hyperparameter optimization
- **Performance Tracking**: Comprehensive metrics including MAE, RMSE, and R² score
- **Web Interface**: User-friendly Flask application for real-time predictions

### 🛡️ **Quality Assurance**
- **Custom Logging**: Comprehensive error tracking and debugging
- **Exception Handling**: Robust error management throughout the pipeline
- **Modular Architecture**: Reusable and maintainable codebase
- **Data Validation**: Input validation and sanitization

---

## 🚀 Installation & Setup

### 📋 Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### 🔧 Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Insurance_Premium_Prediction.git
cd Insurance_Premium_Prediction
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Train the Model
```bash
python src/Insurance/pipeline/training.py
```

**Training Process Includes:**
- Data ingestion and validation
- Feature transformation and encoding
- Model training with cross-validation
- Performance metrics calculation
- Artifact storage and versioning

---

## 💻 Usage

### 🌐 Web Application
```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser to access the prediction interface.

### 📱 User Interface

<div align="center">

![Main Interface](https://github.com/user-attachments/assets/a4bc566b-740c-4282-866a-9dfcb8a52d14)

*Main prediction interface with intuitive form design*

![Results Display](https://github.com/user-attachments/assets/64f78fef-df87-4551-b915-0e4e566c5cf4)

*Results display with detailed premium breakdown*

</div>

### 📝 Sample Input
```json
{
  "age": 45,
  "sex": "male",
  "bmi": 29.8,
  "children": 2,
  "smoker": "yes",
  "region": "southeast"
}
```

### 📊 Sample Output
```
Predicted Premium: ₹27,845.12
Confidence Level: 87%
```

---

## 📈 Model Performance

Our CatBoost model achieves exceptional performance across multiple metrics:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 2,631.18 | Average absolute error in premium prediction |
| **RMSE** | 4,152.67 | Root mean square error for model accuracy |
| **R² Score** | 0.87 | 87% of variance explained by the model |

*Performance metrics are automatically generated and stored in `artifacts/metrics.json`*

---

## 📁 Project Structure

```
Insurance_Premium_Prediction/
├── 📁 artifacts/          # Trained models, preprocessors, metrics
├── 📁 data/               # Raw and processed datasets
├── 📁 notebooks/          # Jupyter notebooks for EDA & experimentation
├── 📁 src/
│   └── 📁 Insurance/      # Core source code package
│       ├── 📁 components/  # Data ingestion, transformation, training
│       ├── 📄 utils.py     # Utility functions
│       ├── 📄 logger.py    # Custom logging implementation
│       └── 📄 exception.py # Custom exception handling
├── 🌐 app.py              # Flask web application
├── 📋 requirements.txt    # Python dependencies
├── ⚙️ setup.py           # Package setup script
├── 📖 README.md          # Project documentation
└── 🚫 .gitignore         # Git ignore patterns
```

---

## 🔮 Future Enhancements

### 🚀 **Planned Features**
- **DVC Integration**: Data version control for pipeline management
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Containerization**: Docker support for consistent deployment
- **Cloud Deployment**: AWS/GCP integration for scalable hosting

### 📊 **Advanced Analytics**
- **Real-time Monitoring**: Model performance tracking and alerting
- **A/B Testing**: Model comparison and selection frameworks
- **API Documentation**: Comprehensive API reference and examples

---

## 🤝 Contributing

We welcome contributions from the community! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### 📋 Contribution Guidelines
- Follow PEP 8 coding standards
- Add comprehensive tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

---

## 📞 Contact

<div align="center">

**Vaidik Yadav**  
*Machine Learning Engineer & Data Scientist*

| Platform | Link |
|----------|------|
| 📧 **Email** | [vaidiky90@gmail.com](mailto:vaidiky90@gmail.com) |
| 💼 **LinkedIn** | [Connect on LinkedIn](https://www.linkedin.com/in/vaidik-yadav-260a60248/) |
| 🐙 **GitHub** | [View Profile](https://github.com/your-username) |

</div>

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

*Built with ❤️ using Python, Machine Learning, and Flask*

</div>
