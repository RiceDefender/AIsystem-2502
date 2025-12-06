import argparse
from pathlib import Path

import onnx
import torch
from torch import nn


class ToyFRModel(nn.Module):
    """
    Minimal CPU-friendly face-embedding backbone.

    This is intentionally light-weight so the Triton pipeline works even if
    students have not yet swapped in a real FR model. Replace with your own
    architecture when ready.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def _load_state_if_available(model: nn.Module, weights_path: Path) -> None:
    if not weights_path.exists():
        print(f"[convert] No weights found at {weights_path}, exporting randomly initialized toy model.")
        return

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
        missing_unexpected = model.load_state_dict(state, strict=False)
        if missing_unexpected.missing_keys:
            print(f"[convert] Missing keys: {missing_unexpected.missing_keys}")
        if missing_unexpected.unexpected_keys:
            print(f"[convert] Unexpected keys: {missing_unexpected.unexpected_keys}")
    else:
        model.load_state_dict(state)
    print(f"[convert] Loaded weights from {weights_path}")


def convert_model_to_onnx(weights_path: Path, onnx_path: Path, opset: int) -> None:
    """
    Convert a (placeholder) FR model to ONNX for Triton CPU serving.

    Swap out `ToyFRModel` with your actual FR backbone once ready; the rest of
    the export pipeline stays the same.
    """
    model = ToyFRModel()
    _load_state_if_available(model, weights_path)
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 112, 112)
    dynamic_axes = {"input": {0: "batch"}, "embedding": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"[convert] ONNX export complete: {onnx_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FR model to ONNX for Triton.")
    parser.add_argument("--weights-path", type=Path, required=True, help="Path to FR model weights (.pth).")
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("model_repository/fr_model/1/model.onnx"),
        help="Destination for exported ONNX file.",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_model_to_onnx(args.weights_path, args.onnx_path, args.opset)


if __name__ == "__main__":
    main()
