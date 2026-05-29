"""
Inference + Grad-CAM explainability service.
Used by the FastAPI backend to run predictions.
"""

import io
import json
import base64
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
from pathlib import Path


# ─── Disease treatment recommendations ────────────────────────────────────────

TREATMENTS = {
    "healthy": {
        "severity": "none",
        "description": "Plant appears healthy. No treatment needed.",
        "actions": ["Continue regular watering schedule", "Monitor weekly for early signs"],
    },
    "default": {
        "severity": "moderate",
        "description": "Disease detected. Treatment recommended.",
        "actions": [
            "Remove and destroy infected leaves immediately",
            "Apply appropriate fungicide/pesticide",
            "Improve air circulation around plants",
            "Avoid overhead irrigation",
            "Consult local agronomist if condition worsens",
        ],
    },
}


class CropDiseasePredictor:
    def __init__(self, model_path: str, metadata_path: str, class_names_path: str, device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load class names & metadata
        with open(class_names_path) as f:
            self.class_names = json.load(f)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.num_classes = len(self.class_names)
        self.img_size = self.metadata.get("img_size", 300)

        # Load model
        self.model = self._build_model(self.metadata["model"])
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval().to(self.device)

        # Grab last conv layer for Grad-CAM
        self._register_hooks()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.gradients = None
        self.activations = None

    def _build_model(self, model_name: str):
        if model_name == "efficientnet_b3":
            model = models.efficientnet_b3()
            model.classifier[1] = torch.nn.Linear(
                model.classifier[1].in_features, self.num_classes
            )
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0()
            model.classifier[1] = torch.nn.Linear(
                model.classifier[1].in_features, self.num_classes
            )
        elif model_name == "resnet50":
            model = models.resnet50()
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def _register_hooks(self):
        """Register forward/backward hooks on the last conv layer for Grad-CAM."""
        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # EfficientNet: features[-1], ResNet: layer4
        if hasattr(self.model, "features"):
            target_layer = self.model.features[-1]
        else:
            target_layer = self.model.layer4

        target_layer.register_forward_hook(save_activation)
        target_layer.register_full_backward_hook(save_gradient)

    def _generate_gradcam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for given class."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        # Global average pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(self.img_size, self.img_size),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def _overlay_heatmap(self, original_img: Image.Image, cam: np.ndarray) -> str:
        """Overlay Grad-CAM on original image, return as base64 PNG."""
        img_array = np.array(original_img.resize((self.img_size, self.img_size)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
        pil_overlay = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        """
        Run inference on a PIL image.
        Returns prediction, confidence, top-k, Grad-CAM heatmap, and treatment.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = probs.topk(top_k)
        predicted_class = self.class_names[top_indices[0].item()]
        confidence = top_probs[0].item()

        # Parse class name: "Tomato___Early_blight" → plant + disease
        parts = predicted_class.split("___")
        plant = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

        # Grad-CAM
        cam = self._generate_gradcam(input_tensor, top_indices[0].item())
        heatmap_b64 = self._overlay_heatmap(image, cam)

        # Treatment recommendation
        is_healthy = "healthy" in predicted_class.lower()
        treatment = TREATMENTS["healthy"] if is_healthy else {
            **TREATMENTS["default"],
            "description": f"{disease} detected on {plant}.",
        }

        return {
            "plant": plant,
            "disease": disease,
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "is_healthy": is_healthy,
            "top_predictions": [
                {
                    "class": self.class_names[idx.item()],
                    "confidence": round(prob.item() * 100, 2),
                }
                for prob, idx in zip(top_probs, top_indices)
            ],
            "gradcam_heatmap": heatmap_b64,
            "treatment": treatment,
            "needs_expert_review": confidence < 0.80,
        }
