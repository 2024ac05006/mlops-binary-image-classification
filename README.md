# 🐱🐶 Cats vs. Dogs — MLOps Binary Image Classification

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13ADC7.svg)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5.svg)](https://kubernetes.io/)
[![Jenkins](https://img.shields.io/badge/Jenkins-CI%2FCD-D24939.svg)](https://www.jenkins.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800.svg)](https://grafana.com/)
[![pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC.svg)](https://pytest.org/)

> **An end-to-end, production-oriented MLOps pipeline for Cats vs. Dogs binary image classification.**

This project demonstrates the complete machine-learning lifecycle: **data versioning → preprocessing → model training → experiment tracking → containerization → CI/CD → Kubernetes deployment → API serving → monitoring and observability**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#1-prerequisites)
  - [Environment Setup](#2-environment-setup)
  - [Data Versioning with DVC](#3-data-versioning-with-dvc)
- [Experimentation and Training](#-experimentation-and-training)
  - [MLflow](#start-mlflow-tracking-server)
  - [Baseline Training](#run-baseline-model-training)
  - [Hyperparameter Experiments](#run-multiple-experiments)
- [Containerization and Local Serving](#-containerization-and-local-serving)
- [Kubernetes Deployment](#-kubernetes-deployment)
- [Monitoring and Observability](#-monitoring-and-observability)
- [API Usage](#-api-usage)
- [Testing](#-testing)
- [CI/CD Pipeline](#-cicd-pipeline)
- [End-to-End MLOps Workflow](#-end-to-end-mlops-workflow)
- [Future Enhancements](#-future-enhancements)
- [Repository](#-repository)

---

## 🔎 Overview

The pipeline is designed around reproducibility and operational readiness.

### Machine Learning

- Binary image classification for **Cats vs. Dogs**
- Modular **PyTorch `SimpleCNN`** training architecture
- Image preprocessing and augmentation
- Dynamic hyperparameter execution
- Loss curves and confusion-matrix artifacts
- Serialized PyTorch model weights (`.pt`)

### MLOps

- **DVC** for reproducible dataset versioning
- **MLflow** for experiment tracking and artifact management
- **Docker** for reproducible model-serving environments
- **Jenkins** for automated CI/CD
- **GitHub Container Registry (GHCR)** for image delivery
- **Kubernetes** for resilient application deployment
- **Prometheus + Grafana** for operational observability

### Serving

The trained model is exposed through a **FastAPI** inference service with:

- `/predict` — model inference
- `/metrics` — Prometheus-compatible metrics
- `/docs` — interactive Swagger API documentation

---

# 🏗️ System Architecture

```text
===================================================================================
                         END-TO-END MLOPS ARCHITECTURE
===================================================================================

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    1. ENGINEERING & EXPERIMENTATION                             │
└─────────────────────────────────────────────────────────────────────────────────┘

                         Local Workstation
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
┌──────────────────────────┐           ┌──────────────────────────┐
│ Data Preprocessing       │           │ PyTorch Training         │
│ data/processed/          │           │ src/train.py             │
│ train / val / test       │           │                          │
│ DVC Versioned            │           │ SimpleCNN                │
└──────────────────────────┘           └────────────┬─────────────┘
                                                     │
                              ┌──────────────────────┴──────────────────────┐
                              │                                             │
                              ▼                                             ▼
                   ┌────────────────────┐                         ┌────────────────────┐
                   │ MLflow Server      │                         │ Model Artifact     │
                   │ :5000              │                         │ baseline_cnn.pt    │
                   │ Params / Metrics   │                         │                    │
                   │ Plots / Artifacts  │                         │ Serialized Weights │
                   └────────────────────┘                         └────────────────────┘

                                │ Git Push
                                ▼

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         2. CI/CD AUTOMATION — JENKINS                           │
└─────────────────────────────────────────────────────────────────────────────────┘

     Checkout SCM
          │
          ▼
  Install Dependencies
          │
          ▼
     Unit Tests
       pytest
          │
          ▼
    Build Docker Image
          │
          ▼
      Publish GHCR
          │
          ▼
   Deploy to Kubernetes
          │
          ▼
      Smoke Tests
          │
          ▼
    Pipeline Passed

                                │ kubectl apply
                                ▼

┌─────────────────────────────────────────────────────────────────────────────────┐
│                   3. KUBERNETES & OBSERVABILITY                                │
└─────────────────────────────────────────────────────────────────────────────────┘

                        Kubernetes Cluster
                                │
                       ┌────────┴────────┐
                       │                 │
                       ▼                 ▼
              ┌────────────────┐  ┌──────────────────┐
              │ FastAPI Service│  │ Monitoring Stack │
              │     :8000      │  │                  │
              └───────┬────────┘  └────────┬─────────┘
                      │                    │
               ┌──────┴──────┐             │
               ▼             ▼             ▼
           ┌───────┐     ┌───────┐   ┌─────────────┐
           │ Pod A │     │ Pod B │   │ Prometheus  │
           │FastAPI│     │FastAPI│   │    :9090    │
           └───────┘     └───────┘   └──────┬──────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │   Grafana   │
                                      │    :3000    │
                                      └─────────────┘
```

---

# ✨ Key Features

| Area | Capability |
|---|---|
| **Data** | Structured `train`, `val`, and `test` datasets |
| **Versioning** | DVC-based dataset versioning |
| **Training** | PyTorch `SimpleCNN` with augmentation |
| **Experiments** | Dynamic hyperparameter execution |
| **Tracking** | MLflow parameters, metrics, plots, and model artifacts |
| **Serving** | FastAPI inference API |
| **Telemetry** | Custom Prometheus metrics |
| **Containerization** | Multi-stage Docker image |
| **CI/CD** | Jenkins automated pipeline |
| **Registry** | GitHub Container Registry |
| **Orchestration** | Kubernetes multi-replica deployment |
| **Monitoring** | Prometheus + Grafana |
| **Testing** | pytest unit/integration tests |
| **Load Testing** | Synthetic random-image request generator |

---

# 🧰 Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11+ |
| Deep Learning | PyTorch |
| Image Processing | PyTorch / Image Processing Libraries |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Containerization | Docker |
| CI/CD | Jenkins |
| Container Registry | GitHub Container Registry |
| Orchestration | Kubernetes |
| Metrics | Prometheus |
| Visualization | Grafana |
| Testing | pytest |
| Version Control | Git / Git LFS |

---

# 📂 Repository Structure

```text
mlops-binary-image-classification/
├── data/
│   └── processed/                  # Preprocessed dataset (train, val, test)
│
├── k8s/                            # Kubernetes deployment and monitoring manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── prometheus.yaml
│   └── grafana.yaml
│
├── models/                         # Serialized model artifacts
│
├── scripts/
│   ├── generate_eda_plots.py       # Feature extraction and EDA heatmaps
│   ├── run_experiments.py         # Hyperparameter grid execution
│   └── send_random_requests.py    # Synthetic telemetry/load generator
│
├── src/
│   ├── app.py                     # FastAPI application + Prometheus instrumentation
│   ├── data_preprocess.py         # Image preprocessing routines
│   └── train.py                   # PyTorch training + MLflow logging
│
├── tests/                          # Unit and integration tests
│
├── Dockerfile                      # Multi-stage container definition
├── Jenkinsfile                     # Jenkins CI/CD pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

# 🚀 Getting Started

## 1. Prerequisites

Ensure the following tools are installed:

- **Python 3.11+**
- **Docker Desktop** with Kubernetes enabled
- **Git**
- **Git LFS**
- **DVC**
- **kubectl**

Verify the installations:

```bash
python --version
docker --version
git --version
dvc --version
kubectl version --client
```

---

## 2. Environment Setup

Clone the repository:

```bash
git clone https://github.com/2024ac05006/mlops-binary-image-classification.git
cd mlops-binary-image-classification
```

Create a virtual environment:

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Data Versioning with DVC

Pull the version-controlled dataset:

```bash
dvc pull
```

The expected processed dataset structure is:

```text
data/processed/
├── train/
├── val/
└── test/
```

DVC allows the dataset version used for an experiment or model build to be reproduced consistently.

---

# 🔬 Experimentation and Training

## Start MLflow Tracking Server

Start MLflow locally:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Open the MLflow dashboard:

```text
http://localhost:5000
```

MLflow is used to track:

- Hyperparameters
- Training and validation metrics
- Loss curves
- Confusion matrices
- Model artifacts
- Experiment runs

---

## Run Baseline Model Training

Train the baseline PyTorch CNN:

```bash
python -m src.train
```

The training pipeline produces serialized model weights such as:

```text
models/baseline_cnn.pt
```

Training artifacts can also be logged to MLflow.

---

## Run Multiple Experiments

Execute the hyperparameter experiment workflow:

```bash
python scripts/run_experiments.py
```

This allows multiple training configurations to be evaluated and compared through MLflow.

---

# 🐳 Containerization and Local Serving

## Build Docker Image

Build the model-serving image:

```bash
docker build -t cats-dogs-service:latest .
```

Verify the image:

```bash
docker images
```

---

## Run the Container

Start the FastAPI service:

```bash
docker run -d \
  -p 8000:8000 \
  --name mlops-api \
  cats-dogs-service:latest
```

Check the running container:

```bash
docker ps
```

---

## Access the API

Open the interactive Swagger documentation:

```text
http://localhost:8000/docs
```

The service exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Classify an uploaded image |
| `/metrics` | GET | Prometheus telemetry |
| `/docs` | GET | Interactive Swagger UI |

---

# ☸️ Kubernetes Deployment

The project includes Kubernetes manifests for the application and observability stack.

## Deploy the Application

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Deploy Prometheus and Grafana:

```bash
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml
```

---

## Verify Deployment

Check pods:

```bash
kubectl get pods
```

Check services:

```bash
kubectl get services
```

Check deployments:

```bash
kubectl get deployments
```

For more detailed troubleshooting:

```bash
kubectl describe pods
```

---

# 🔌 Port Forwarding

Port forwarding allows the Kubernetes services to be accessed from the local workstation.

## FastAPI

```bash
kubectl port-forward svc/cats-dogs-service 8000:8000
```

Access:

```text
http://localhost:8000/docs
```

---

## Prometheus

```bash
kubectl port-forward svc/prometheus-service 9090:9090
```

Access:

```text
http://localhost:9090
```

---

## Grafana

```bash
kubectl port-forward svc/grafana-service 3000:3000
```

Access:

```text
http://localhost:3000
```

> **Note:** Port-forward commands are normally kept running in their own terminal sessions.

---

# 📊 Monitoring and Observability

The FastAPI application exposes Prometheus-compatible metrics through:

```text
GET /metrics
```

A custom prediction counter is exposed as:

```text
model_predictions_total
```

The observability flow is:

```text
┌──────────────────┐
│ FastAPI /predict │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Application      │
│ Metrics          │
│ /metrics         │
└────────┬─────────┘
         │ scrape
         ▼
┌──────────────────┐
│ Prometheus       │
│ :9090            │
└────────┬─────────┘
         │ query
         ▼
┌──────────────────┐
│ Grafana          │
│ :3000            │
└──────────────────┘
```

### Metrics and telemetry can be used to monitor:

- Request throughput
- Prediction counts
- Prediction distribution
- API request latency
- Application health
- Kubernetes service behavior

---

# 🧪 Load Testing

Generate synthetic inference traffic:

```bash
python scripts/send_random_requests.py
```

The script selects **30 random images** from:

```text
data/processed/test/
```

and sends them to the `/predict` endpoint.

This generates live application telemetry that can be observed through Prometheus and Grafana.

---

# 📈 Dashboard Access

| Component | Address | Purpose |
|---|---|---|
| **FastAPI Swagger** | `http://localhost:8000/docs` | API testing |
| **FastAPI Metrics** | `http://localhost:8000/metrics` | Application telemetry |
| **MLflow** | `http://localhost:5000` | Experiment tracking |
| **Prometheus** | `http://localhost:9090` | Metrics collection |
| **Grafana** | `http://localhost:3000` | Monitoring dashboards |

### Grafana Default Credentials

```text
Username: admin
Password: admin
```

> ⚠️ **Security:** Change the default Grafana password before using this configuration outside a local/demo environment.

---

# 🔌 API Usage

## Interactive API Documentation

FastAPI automatically provides Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Prediction Endpoint

```text
POST /predict
```

The endpoint accepts an image and returns the model's binary classification prediction.

A typical workflow is:

```text
Image
  │
  ▼
POST /predict
  │
  ▼
Image Preprocessing
  │
  ▼
PyTorch Model
  │
  ▼
Prediction
  │
  ▼
JSON Response
```

---

# 🧪 Testing

Run the complete test suite:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

The tests provide automated validation of API and model-related functionality before containerization and deployment.

---

# 🔄 CI/CD Pipeline

The `Jenkinsfile` defines the automated CI/CD workflow.

The pipeline is triggered by repository changes and performs:

```text
Git Push
   │
   ▼
Checkout SCM
   │
   ▼
Install Dependencies
   │
   ▼
Run Unit Tests
   │
   ▼
Build Docker Image
   │
   ▼
Publish to GHCR
   │
   ▼
Deploy to Kubernetes
   │
   ▼
Run Smoke Tests
   │
   ▼
Pipeline Passed
```

---

## 🛠️ Jenkins Pipeline Stages

| Stage | Description | Average Execution Time |
|---|---|---:|
| **Checkout SCM** | Clones source repository and fetches branch state | ~3s |
| **Install Dependencies** | Creates environment and resolves Python dependencies | ~4m 20s |
| **Run Unit Tests** | Executes the `pytest` test suite | ~25s |
| **Build Docker Image** | Builds multi-stage container with model weights | ~2m 25s |
| **Publish to GHCR** | Tags and pushes image to GitHub Container Registry | ~15s |
| **Deploy to K8s** | Updates Kubernetes deployment | ~1m 30s |
| **Smoke Tests** | Validates post-deployment endpoint availability | — |

> **Execution times are representative averages and may vary depending on the Jenkins agent, network, dependency cache, and cluster environment.**

---

# 📦 Docker Image Lifecycle

```text
                    ┌─────────────────┐
                    │   Source Code   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Docker Build  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Local Image   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      GHCR       │
                    │ Container Image │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Kubernetes    │
                    │   Deployment    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ FastAPI Pods    │
                    │ Model Inference │
                    └─────────────────┘
```

---

# ☸️ Kubernetes Architecture

The Kubernetes deployment provides:

- Multiple application replicas
- Service-based traffic routing
- ConfigMap-based configuration
- Pod health checks
- Rolling deployment support
- Independent monitoring components

Conceptual architecture:

```text
                         Kubernetes Cluster
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
        ┌──────────────────┐        ┌──────────────────┐
        │  FastAPI Service │        │ Monitoring Stack │
        │      :8000       │        │                  │
        └────────┬─────────┘        └────────┬─────────┘
                 │                           │
           ┌─────┴─────┐              ┌─────┴─────┐
           ▼           ▼              ▼           ▼
       ┌───────┐   ┌───────┐    ┌──────────┐ ┌─────────┐
       │ Pod A │   │ Pod B │    │Prometheus│ │ Grafana │
       │FastAPI│   │FastAPI│    │  :9090   │ │  :3000  │
       └───────┘   └───────┘    └──────────┘ └─────────┘
           │           │
           └─────┬─────┘
                 │
                 ▼
          Model Inference
```

---

# 🔁 End-to-End MLOps Workflow

```text
┌─────────────────────┐
│    Raw Dataset      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Preprocessing  │
│    + DVC Versioning │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ PyTorch Training    │
│     SimpleCNN       │
└──────────┬──────────┘
           │
           ├──────────────► MLflow Tracking
           │
           ▼
┌─────────────────────┐
│ Model Artifact      │
│     .pt             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Docker Packaging    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Jenkins CI/CD       │
│ Test → Build → Push │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Kubernetes Deploy   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FastAPI Inference   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Prometheus Metrics  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Grafana Dashboards  │
└─────────────────────┘
```

---

# 🎯 MLOps Lifecycle Coverage

This project covers the major stages of an MLOps lifecycle:

| Lifecycle Stage | Implementation |
|---|---|
| **Data Management** | DVC |
| **Data Preparation** | `src/data_preprocess.py` |
| **Experimentation** | MLflow |
| **Model Training** | PyTorch |
| **Model Artifact Management** | `.pt` artifacts |
| **Testing** | pytest |
| **Packaging** | Docker |
| **Continuous Integration** | Jenkins |
| **Artifact Delivery** | GHCR |
| **Deployment** | Kubernetes |
| **Model Serving** | FastAPI |
| **Metrics Collection** | Prometheus |
| **Visualization** | Grafana |
| **Operational Testing** | Smoke tests + synthetic load |

---

# 🏁 Conclusion

This project demonstrates a complete **production-oriented MLOps lifecycle for image classification**, from reproducible dataset management and model experimentation through automated delivery, Kubernetes-based model serving, and real-time operational monitoring.

The architecture provides a practical foundation for extending a machine-learning prototype into a more robust production system.

