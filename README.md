# 🛂 US Visa Approval Prediction — MLOps Production ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Cloud-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API%20Server-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

This is a **production-ready MLOps project** that builds a **US Visa Approval Prediction** system. The model predicts whether a US Visa application will be **certified or denied**, based on applicant profile data.

The project is designed for real-world deployment with **MongoDB** as the data store, **AWS S3** for model registry, **FastAPI** for serving, and **Docker** for containerisation — following complete MLOps best practices.

---

## 🎯 Business Problem

Immigration authorities process thousands of visa applications daily. This model helps recommend approval or denial based on structured criteria, enabling faster and more consistent decision-making.

> "Recommend a suitable profile for applicants for whom the visa should be certified or denied based on criteria that influence the decision."

---

## 🏗️ Architecture & Workflow

### Pipeline Components

```
1. constants          → Static configuration values
2. entity             → Data class definitions (config & artifact entities)
3. components         → Core pipeline logic
4. pipeline           → Stage orchestration
5. main.py            → Full pipeline runner
6. app.py             → FastAPI web server
```

### MLOps Stack

| Component | Tool |
|-----------|------|
| Database | MongoDB Atlas |
| Cloud Storage | AWS S3 (model registry) |
| API Server | FastAPI |
| Container | Docker |
| ML Monitoring | Evidently AI |
| Flowchart Design | Whimsical |

---

## 📁 Project Structure

```
MLOPs-Production-Ready-Machine-Learning-Project/
│
├── us_visa/                  # Core ML package
│   ├── components/           # Data ingestion, validation, transformation, training
│   ├── pipeline/             # Training & prediction pipelines
│   ├── entity/               # Config & artifact data classes
│   └── constants/            # Global constants
│
├── notebook/                 # EDA and research notebooks
├── config/                   # YAML configuration files
├── static/                   # Static web assets
├── templates/                # HTML templates (Jinja2)
├── app.py                    # FastAPI application
├── demo.py                   # Demo runner script
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container config
├── setup.py                  # Package installer
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Scikit-learn | ML modelling |
| MongoDB | Data storage |
| AWS S3 | Model artifact registry |
| FastAPI | REST API serving |
| Docker | Containerised deployment |
| Evidently AI | ML monitoring & drift detection |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/homeshwarnelakurthi/MLOPs-Production-Ready-Machine-Learning-Project.git
cd MLOPs-Production-Ready-Machine-Learning-Project
```

### 2. Create a virtual environment
```bash
conda create -n visa python=3.8 -y
conda activate visa
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net"
export AWS_ACCESS_KEY_ID=<your_key_id>
export AWS_SECRET_ACCESS_KEY=<your_secret_key>
```

### 5. Run the application
```bash
python app.py
```

### 6. Run with Docker
```bash
docker build -t us-visa-predictor .
docker run -p 8080:8080 us-visa-predictor
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page |
| `POST` | `/predict` | Predict visa approval |
| `GET` | `/train` | Trigger training pipeline |

---

## 👨‍💻 Author

**Homeswar Rao Nelakurthi**  
[![GitHub](https://img.shields.io/badge/GitHub-homeshwarnelakurthi-181717?style=flat&logo=github)](https://github.com/homeshwarnelakurthi)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
