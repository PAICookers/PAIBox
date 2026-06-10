"""Deploy quantized ResNetCIFAR10 through PAIIR and backendv2."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_fx
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent.parent.parent
for path in (REPO_ROOT, BASE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from paibox.backendv2.mapper import Mapper  # noqa: E402
from paibox.paiir import compile_to_paiir  # noqa: E402
from paibox.quantize_tools import (  # noqa: E402
    build_manual_prepare_custom_config,
    build_bked_backend_config,
    build_bked_qconfig_mapping,
    convert_fx_to_manual,
    export_manual_model_params,
    export_quantized_model_summary,
    register_manual_quantized_paiir,
)
from model import ResNetCIFAR10  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "examples" / "data"
DEFAULT_CKPT = BASE_DIR / "checkpoints" / "best_model.pth"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"


def build_calibration_loader(
    data_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    download: bool,
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
        train=True,
        download=download,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )


def build_test_loader(
    data_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    download: bool,
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
        train=False,
        download=download,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_fp32_model(checkpoint_path: Path) -> ResNetCIFAR10:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ResNetCIFAR10(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def calibrate(model: nn.Module, loader: DataLoader, device: torch.device, batches: int) -> None:
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= batches:
                break
            model(images.to(device))


def compare_model_accuracy(
    fp32_model: nn.Module,
    hardware_sim_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    batches: int,
) -> tuple[float, float, float, int]:
    fp32_model.eval()
    hardware_sim_model.eval()

    fp32_correct = 0
    hardware_correct = 0
    agreement = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if batches > 0 and i >= batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            fp32_pred = fp32_model(images).argmax(dim=1)
            hardware_pred = hardware_sim_model(images).argmax(dim=1)

            fp32_correct += fp32_pred.eq(labels).sum().item()
            hardware_correct += hardware_pred.eq(labels).sum().item()
            agreement += fp32_pred.eq(hardware_pred).sum().item()
            total += labels.numel()

    if total == 0:
        raise ValueError("accuracy comparison received no samples")

    fp32_acc = 100.0 * fp32_correct / total
    hardware_acc = 100.0 * hardware_correct / total
    agreement_acc = 100.0 * agreement / total
    return fp32_acc, hardware_acc, agreement_acc, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize and deploy ResNetCIFAR10 through PAIIR/backendv2."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calib-batches", type=int, default=10)
    parser.add_argument(
        "--accuracy-batches",
        type=int,
        default=10,
        help="Number of test batches for FP32 vs hardware-sim accuracy comparison; <=0 runs all.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--target-platform",
        choices=("all", "x86", "riscv"),
        default="all",
    )
    parser.add_argument("--no-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== ResNetCIFAR10 PAIIR/backendv2 deployment ===")
    print(f"[1] Loading checkpoint: {args.checkpoint}")
    fp32_model = load_fp32_model(args.checkpoint).to(device)
    sample_input = torch.randn(1, 3, 32, 32).to(device)

    print("[2] FX fuse and prepare with BKED backend config")
    backend_config = build_bked_backend_config()
    fused_model = fuse_fx(fp32_model, backend_config=backend_config)
    prepared_model = prepare_fx(
        fused_model,
        build_bked_qconfig_mapping(symmetric=True),
        (sample_input,),
        prepare_custom_config=build_manual_prepare_custom_config(),
        backend_config=backend_config,
    )

    print(f"[3] Calibrating with {args.calib_batches} batch(es)")
    calibration_loader = build_calibration_loader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    calibrate(prepared_model, calibration_loader, device, args.calib_batches)

    print("[4] Converting FX graph to manual quantized modules")
    quantized_model = convert_fx_to_manual(
        prepared_model,
        backend_config=backend_config,
    )
    quantized_model.graph.print_tabular()

    print("[5] Exporting quantized summaries and int weights")
    export_quantized_model_summary(
        quantized_model,
        str(output_dir / "quantized_model_layers.json"),
        str(output_dir / "quantized_model_graph.txt"),
    )
    params_dir = output_dir / "exported_params"
    params_dir.mkdir(parents=True, exist_ok=True)
    export_manual_model_params(quantized_model, params_dir)

    print("[5.5] Comparing FP32 and hardware-sim model accuracy")
    test_loader = build_test_loader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    fp32_acc, hardware_acc, agreement_acc, eval_samples = compare_model_accuracy(
        fp32_model,
        quantized_model,
        test_loader,
        device,
        args.accuracy_batches,
    )
    print(f"      Samples: {eval_samples}")
    print(f"      FP32 accuracy: {fp32_acc:.2f}%")
    print(f"      Hardware-sim accuracy: {hardware_acc:.2f}%")
    print(f"      Accuracy delta: {hardware_acc - fp32_acc:+.2f}%")
    print(f"      Prediction agreement: {agreement_acc:.2f}%")

    print("[6] Compiling to PAIIR")
    register_manual_quantized_paiir()
    graph = compile_to_paiir(quantized_model, sample_input)
    graph.summary(True)

    print("[7] Running backendv2 mapper")
    mapper = Mapper()
    mapper.compile(
        graph,
        output_path=output_dir / "frame_out",
        target_platform=args.target_platform,
        debug=not args.no_debug,
    )

    print(f"[Done] Output directory: {output_dir.resolve()}")
    print(f"[Done] Routing groups: {len(mapper.routing_groups)}")
    print(f"[Done] Core placements: {len(mapper.coreplacements)}")


if __name__ == "__main__":
    main()
