"""Quantize ResNetCIFAR10 with the custom BKED FX backend."""

import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_fx
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
for path in (REPO_ROOT, BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from simples.quantize_tools import (  # noqa: E402
    build_bked_backend_config,
    build_bked_qconfig_mapping,
    convert_fx_to_manual,
    export_manual_model_params,
    export_quantized_model_summary,
)
from simples.res_cifar10.model import ResNetCIFAR10  # noqa: E402


DATA_DIR = os.path.join(REPO_ROOT, "simples", "data")
SOURCE_CKPT = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_custom_backend")
MODEL_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "quantized_model_layers.json")
MODEL_GRAPH_TXT = os.path.join(OUTPUT_DIR, "quantized_model_graph.txt")
EXPORT_PARAMS_DIR = os.path.join(OUTPUT_DIR, "exported_params")


def build_loader(
    data_dir: str,
    train: bool,
    batch_size: int = 256,
    num_workers: int = 2,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_fp32_model() -> ResNetCIFAR10:
    if not os.path.exists(SOURCE_CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {SOURCE_CKPT}")

    model = ResNetCIFAR10(num_classes=10)
    checkpoint = torch.load(SOURCE_CKPT, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def print_fused_modules(model: torch.fx.GraphModule) -> None:
    print("Fused modules from custom backend:")
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in {"ConvReLU2d", "ConvAddReLU2d", "LinearReLU"}:
            print(f"  {name}: {module_type}")


def main() -> None:
    device = torch.device("cpu")
    symmetric = True
    calib_batches = 10
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== ResNetCIFAR10 custom-backend FX quantization ===")
    fp32_model = load_fp32_model().to(device)
    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    backend_config = build_bked_backend_config()
    qconfig_mapping = build_bked_qconfig_mapping(symmetric=symmetric)

    fused_model = fuse_fx(fp32_model, backend_config=backend_config)
    print_fused_modules(fused_model)

    prepared_model = prepare_fx(
        fused_model,
        qconfig_mapping,
        (example_inputs,),
        backend_config=backend_config,
    )

    calib_loader = build_loader(DATA_DIR, train=True)
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= calib_batches:
                break
            prepared_model(images.to(device))

    quantized_model = convert_fx_to_manual(
        prepared_model,
        activation_symmetric=symmetric,
    )

    test_loader = build_loader(DATA_DIR, train=False)
    fp32_acc = evaluate_model(fp32_model, test_loader, device)
    quantized_acc = evaluate_model(quantized_model, test_loader, device)
    print(f"FP32 accuracy: {fp32_acc:.2f}%")
    print(f"Quantized accuracy: {quantized_acc:.2f}%")

    export_quantized_model_summary(
        quantized_model,
        MODEL_SUMMARY_JSON,
        MODEL_GRAPH_TXT,
    )
    os.makedirs(EXPORT_PARAMS_DIR, exist_ok=True)
    export_manual_model_params(quantized_model, EXPORT_PARAMS_DIR)
    print(f"Summary JSON: {MODEL_SUMMARY_JSON}")
    print(f"Graph TXT: {MODEL_GRAPH_TXT}")
    print(f"Exported params: {EXPORT_PARAMS_DIR}")


if __name__ == "__main__":
    main()
