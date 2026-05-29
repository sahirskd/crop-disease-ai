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

## 🚀 Local Development & Debugging

### 1. Dataset Setup
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

### 3. Backend (Local Run)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

### 4. Frontend (Local Run)
```bash
cd frontend
npm install
npm run dev
# App at http://localhost:5173
```

### 5. Debugging with VS Code
The project includes a `.vscode/launch.json` file for easy debugging.
1. Open the project root in VS Code.
2. Go to the Run and Debug view (`Cmd+Shift+D` or `Ctrl+Shift+D`).
3. Select **"Debug FastAPI"** from the dropdown to debug the backend.
4. Select **"Debug React Frontend"** to debug the frontend in Chrome.

## 🚀 Production Deployment (Docker)

The application uses multi-stage Docker builds for optimized production deployment.
- **Backend**: Uses a non-root user and runs FastAPI via Gunicorn with Uvicorn workers.
- **Frontend**: Compiles the React application and serves static assets via an Nginx alpine container.

To deploy the full stack in production:
```bash
docker-compose up --build -d
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
