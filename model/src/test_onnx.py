import onnxruntime as ort
import numpy as np
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import onnxruntime as ort

def get_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def main():
    # Load model and weights
    model_path = "experiments/edge/cropsense_fp32.onnx"
    weights_path = "experiments/edge/classifier_weights.json"
    
    with open(weights_path, "r") as f:
        classifier_weights = np.array(json.load(f)) # [38, 1536]
        
    session = ort.InferenceSession(model_path)
    
    # Create dummy image for testing
    dummy_img = torch.randn(1, 3, 300, 300).numpy()
    
    # Run inference
    inputs = {session.get_inputs()[0].name: dummy_img}
    outputs = session.run(None, inputs)
    
    probs = outputs[0]
    fmaps = outputs[1]
    
    # Verify outputs
    pred_class = np.argmax(probs[0])
    print(f"Predicted class index: {pred_class}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Feature maps shape: {fmaps.shape}")
    
    # Compute CAM for predicted class (Simulating what JS will do)
    # fmaps is [1, 1536, H, W]
    # pred_weights is [1536]
    pred_weights = classifier_weights[pred_class]
    
    # Multiply and sum across channels
    # fmaps[0] is [1536, 10, 10]
    fmaps_squeezed = fmaps[0]
    cam = np.zeros((fmaps_squeezed.shape[1], fmaps_squeezed.shape[2]))
    
    for c in range(1536):
        cam += fmaps_squeezed[c] * pred_weights[c]
        
    cam = np.maximum(cam, 0) # ReLU
    
    # Normalize
    cam_min = np.min(cam)
    cam_max = np.max(cam)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    
    print(f"Generated CAM shape: {cam.shape}")
    print(f"CAM min: {np.min(cam):.4f}, max: {np.max(cam):.4f}")
    
    print("✅ ONNX Edge Model verification successful!")

if __name__ == "__main__":
    main()
