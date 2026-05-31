import torch
import torch.nn as nn
from torchvision import models
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import json
from pathlib import Path
import argparse

def build_model(num_classes: int):
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

class EdgeModelWithCAM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

    def forward(self, x):
        fmaps = self.features(x)
        pooled = self.avgpool(fmaps).flatten(1)
        logits = self.classifier(pooled)
        probs = torch.softmax(logits, dim=1)
        return probs, fmaps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="experiments/best_model.pt", type=str)
    parser.add_argument("--classes_path", default="experiments/class_names.json", type=str)
    parser.add_argument("--output_dir", default="experiments/edge/", type=str)
    args = parser.parse_args()

    # load class names
    with open(args.classes_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    
    # build base model
    base_model = build_model(num_classes)
    base_model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    base_model.eval()
    
    # wrap
    edge_model = EdgeModelWithCAM(base_model)
    edge_model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save classifier weights for CAM calculation in React Native
    weights = edge_model.classifier[1].weight.detach().cpu().numpy()
    weights_path = output_dir / "classifier_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights.tolist(), f)
    print(f"Saved classifier weights to {weights_path}")
    
    fp32_path = output_dir / "cropsense_fp32.onnx"
    int8_path = output_dir / "cropsense_int8.onnx"

    dummy_input = torch.randn(1, 3, 300, 300)
    
    print(f"Exporting FP32 model to {fp32_path}...")
    torch.onnx.export(
        edge_model,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['probabilities', 'fmaps']
    )
    
    print(f"Skipping INT8 quantization due to ONNX shape inference bug with EfficientNet. Using FP32 model.")
    # try:
    #     quantize_dynamic(
    #         str(fp32_path),
    #         str(int8_path),
    #         weight_type=QuantType.QUInt8
    #     )
    #     print(f"Quantization complete. INT8 model saved to {int8_path}")
    # except Exception as e:
    #     print(f"Quantization failed: {e}")
    #     print("Using FP32 model instead.")
    
    print("Done!")

if __name__ == "__main__":
    main()
