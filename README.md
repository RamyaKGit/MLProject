# End-to-End Machine Learning Project: Student Performance Indicator

A production-ready End-to-End Machine Learning web application designed to predict student performance (Math Scores) based on demographic and academic input parameters. Built with **Python**, **Flask**, **Scikit-Learn**, **Docker**, and **GitHub Actions CI/CD with AWS ECR**.

---

## 📌 Project Overview

This project implements an end-to-end ML pipeline encompassing:
1. **Data Ingestion**: Reads raw datasets and creates split train/test artifacts.
2. **Data Transformation**: Handles missing values, performs feature scaling (`StandardScaler`), and encodes categorical variables (`OneHotEncoder`).
3. **Model Training & Evaluation**: Evaluates multiple regression models (Linear Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, CatBoost) to select the best-performing model.
4. **Prediction Pipeline**: Exposes a prediction interface wrapping preprocessing and inference logic.
5. **Web Application**: Flask interface for interactive user inputs and real-time predictions.
6. **Containerization & CI/CD**: Fully Dockerized application integrated with GitHub Actions for automated syntax checking, Docker build verification, and automated image deployment to Amazon ECR.

---

## 🛠️ Tech Stack

- **Programming Language**: Python 3.10
- **Web Framework**: Flask
- **Machine Learning**: Scikit-Learn, XGBoost, CatBoost
- **Data Manipulation & Visualization**: Pandas, NumPy, Seaborn, Matplotlib
- **Containerization**: Docker
- **Cloud & CI/CD**: AWS ECR, GitHub Actions, AWS CLI

---

## 📂 Project Directory Structure

```text
mlprojects/
├── .ebextensions/            # AWS Elastic Beanstalk configuration
│   └── python.config
├── .github/
│   └── workflows/
│       └── main.yaml         # GitHub Actions CI/CD Pipeline (Integration + AWS ECR Delivery)
├── artifacts/                # Generated datasets & serialized model artifacts
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── best_model.pkl
├── notebook/                 # Exploratory Data Analysis & Model Training Notebooks
├── src/                      # Modular Python package source code
│   ├── __init__.py
│   ├── components/           # Core pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/             # Prediction and Training execution pipelines
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py          # Custom exception handling with detailed error tracing
│   ├── logger.py             # Custom logging configuration
│   └── utils.py              # Shared utility functions (model evaluation, object serialization)
├── templates/                # HTML templates for Flask Web UI
│   ├── index.html
│   └── home.html
├── app.py                    # Flask application entrypoint
├── Dockerfile                # Container configuration
├── requirements.txt          # Python dependencies
└── setup.py                  # Package configuration
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Git
- Docker (optional for containerized execution)

### 2. Local Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RamyaKGit/MLProject.git
   cd MLProject
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate        # On macOS/Linux
   # or: venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask Web Application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://localhost:8080` (or `http://localhost:8080/predict`).

---

## 🐳 Docker Deployment to Microsoft Azure

### 1. Local Docker Build & Test
1. **Build the Docker Image**:
   ```bash
   docker build -t mlprojects:latest .
   ```

2. **Run the Docker Container Locally**:
   ```bash
   docker run -p 8080:8080 mlprojects:latest
   ```
   Access the application at `http://localhost:8080`.

---

### 2. Pushing Image to Azure Container Registry (ACR) & Deploying via Azure Portal

Assuming your **Azure Container Registry** and **Azure Web App for Containers** have been created in the **Azure Portal**:

#### Step 1: Login to Azure Container Registry
```bash
docker login testdockerrk.azurecr.io
# OR using Azure CLI:
# az acr login --name testdockerrk
```

#### Step 2: Build, Tag & Push Docker Image to ACR
```bash
# 1. Build the local Docker image
docker build -t mlprojects:latest .

# 2. Tag image for Azure Container Registry
docker tag mlprojects:latest testdockerrk.azurecr.io/studentperformance:latest

# 3. Push image to ACR
docker push testdockerrk.azurecr.io/studentperformance:latest
```

#### Step 3: Configure Azure Web App (Azure Portal)
1. Go to your **Azure Web App** in the Azure Portal.
2. Under **Deployment**, navigate to **Deployment Center**.
3. Select **Azure Container Registry** (`testdockerrk`) as the Source.
4. Select your **Registry** (`testdockerrk`), **Image** (`studentperformance`), and **Tag** (`latest`).
5. Under **Environment variables** / **App settings**, add:
   - **Name**: `WEBSITES_PORT`
   - **Value**: `8080`
6. Save and restart the Web App.


---

## ⚙️ Continuous Integration & Continuous Delivery (CI/CD)

The project includes a GitHub Actions workflow defined in `.github/workflows/main.yaml`.

### Workflow Jobs:
1. **Continuous Integration (`integration`)**:
   - Checks out the repository.
   - Sets up Python 3.10.
   - Installs project dependencies.
   - Performs syntax compilation checks across `setup.py` and `app.py`.
   - Tests Docker container creation (`docker build`).

2. **Continuous Delivery (Azure / ECR Pipeline)**:
   - Triggered on push to `main`.
   - Authenticates with Cloud Container Registry (ACR / ECR).
   - Builds, tags, and pushes the production Docker image to the registry.

### Required GitHub Secrets for Azure Deployment:
To configure automated container delivery to **Azure Container Registry (ACR)** and **Azure App Service**, add the following repository secrets (**Settings > Secrets and variables > Actions**):

| Secret Name | Description | Example / Value |
| :--- | :--- | :--- |
| `AZURE_CREDENTIALS` | Azure Service Principal Credentials JSON | output from `az ad sp create-for-rbac` |
| `REGISTRY_USERNAME` | ACR Admin Username | `mlprojectsregistry` |
| `REGISTRY_PASSWORD` | ACR Admin Password | output from `az acr credential show` |
| `AZURE_WEBAPP_NAME` | Azure App Service Name | `mlprojects-app` |


---

## 📊 Features & Model Input Data

The web interface accepts the following input parameters to predict student performance:
- **Gender**: `female`, `male`
- **Race/Ethnicity**: `group A`, `group B`, `group C`, `group D`, `group E`
- **Parental Level of Education**: `bachelor's degree`, `some college`, `master's degree`, `associate's degree`, `high school`, `some high school`
- **Lunch**: `standard`, `free/reduced`
- **Test Preparation Course**: `none`, `completed`
- **Reading Score**: Numerical score (0-100)
- **Writing Score**: Numerical score (0-100)
