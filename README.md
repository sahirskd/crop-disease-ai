# 🌿 CropSense AI — Crop Disease Detection System

A production-grade deep learning application that detects plant diseases from leaf images with **95–99% accuracy**, powered by EfficientNetB3 + Grad-CAM explainability.

## 🏗️ Architecture

```
crop-disease-ai/
├── model/                  # Training pipeline
│   ├── notebooks/          # EDA + training experiments
│   ├── src/                # Reusable training modules
│   └── experiments/        # MLflow / W&B run artifacts
├── backend/                # FastAPI REST API
│   └── app/
│       ├── api/            # Route handlers
│       ├── core/           # Config, settings
│       ├── models/         # Pydantic schemas
│       └── services/       # Inference, GradCAM, DB
└── frontend/               # React web app
    └── src/
        ├── components/     # UI components
        ├── pages/          # Route pages
        ├── hooks/          # Custom hooks
        └── utils/          # API client, helpers
```

## 🚀 Quick Start

### 1. Dataset
Download PlantVillage from Kaggle:
```bash
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d model/data/
```

### 2. Train the Model
```bash
cd model
pip install -r requirements.txt
python src/train.py --epochs 30 --batch_size 32 --model efficientnet_b3
```

### 3. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

### 4. Frontend
```bash
cd frontend
npm install
npm run dev
# App at http://localhost:5173
```

### 5. Docker (Full Stack)
```bash
docker-compose up --build
```

## 🧠 Model Details
- **Architecture:** EfficientNetB3 (pretrained on ImageNet, fine-tuned)
- **Dataset:** PlantVillage — 54,306 images, 38 disease classes
- **Accuracy:** ~98% validation accuracy
- **Explainability:** Grad-CAM heatmaps per prediction
- **Experiment Tracking:** Weights & Biases (wandb)

## 📡 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | Upload image, get prediction + heatmap |
| GET | `/api/v1/diseases` | List all 38 disease classes |
| GET | `/api/v1/history` | Prediction history |
| GET | `/health` | Health check |

## 🌐 Deployment
- **Model API:** Hugging Face Spaces (Docker SDK)
- **Frontend:** Vercel
- **DB:** Supabase (PostgreSQL)
