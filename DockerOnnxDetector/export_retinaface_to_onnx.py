import argparse
import re
from pathlib import Path

import onnx
import torch

from nms import nms  # noqa: F401  # placeholder; intentionally left empty
from retinaface_model import RetinaFace


WEIGHT_URL = "./retinaface_mv1_0.25.pth"


def download_weights(dest: Path) -> Path:
    """Check for weights locally. Fail if not found and URL is invalid."""
    if dest.exists():
        print(f"Using existing weights: {dest}")
        return dest

    # If code reaches here, it means the file was NOT found at 'dest'
    print(f"[ERROR] Could not find weights at: {dest.resolve()}")

    # specific check to stop the crash you are seeing
    if not WEIGHT_URL.startswith("http"):
        raise FileNotFoundError(
            f"The weights file was not found at '{dest}', and the WEIGHT_URL is not a valid download link. "
            "Please verify your --weights-path argument."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading RetinaFace MobileNet0.25 weights to {dest} ...")
    torch.hub.download_url_to_file(WEIGHT_URL, dest)
    print("Download complete.")
    return dest


def fuse_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b):
    """Fuses BatchNorm parameters into Convolution parameters."""
    eps = 1e-5
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    std = (bn_rv + eps).sqrt()
    t = bn_w / std

    fused_w = conv_w * t.reshape(-1, 1, 1, 1)
    fused_b = bn_b + (conv_b - bn_rm) * t
    return fused_w, fused_b


def adapt_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remaps state_dict keys from the retinaface_mv1_0.25.pth checkpoint."""
    adapted_state = {}
    fusion_buffer = {}

    for k, v in state_dict.items():
        # --- MobileNetV1 Body Mapping (from fx trace) ---
        if k.startswith("fx.stage"):
            parts = k.split('.')
            stage_num = int(parts[1][-1])
            block_idx = int(parts[2])
            
            if stage_num == 1 and block_idx == 0:
                new_k = f"body.conv1.{parts[3]}.{parts[4]}"
            else:
                if stage_num == 1: target_idx = block_idx - 1
                elif stage_num == 2: target_idx = block_idx + 5
                else: target_idx = block_idx + 11
                
                dw_or_pw = parts[3]
                conv_or_bn = parts[4]
                param = parts[5]
                
                if dw_or_pw == '0': # Depthwise part
                    new_k = f"body.stages.{target_idx}.conv.{conv_or_bn}.{param}"
                else: # Pointwise part
                    new_k = f"body.stages.{target_idx}.conv.{int(conv_or_bn) + 3}.{param}"
            
            adapted_state[new_k] = v
            continue

        # --- FPN/SSH Layer Fusion ---
        fusion_match = re.match(r"((?:fpn\.(?:output|merge)\d|ssh\d\.(?:conv3X3|conv5X5_1|conv5X5_2|conv7X7_2|conv7x7_3)))\.(0|1)\.(.*)", k)
        if fusion_match:
            base_name, layer_type, param = fusion_match.groups()
            
            base_name = base_name.replace("conv3X3", "conv3x3")
            base_name = base_name.replace("conv5X5", "conv5x5")
            base_name = base_name.replace("conv7X7", "conv7x7")
            base_name = base_name.replace("conv7x7_3", "conv7x7_2")

            if base_name not in fusion_buffer:
                fusion_buffer[base_name] = {}

            key_name = "conv" if layer_type == "0" else "bn"
            fusion_buffer[base_name][f"{key_name}_{param}"] = v
            continue

        # --- Head layers ---
        new_k = k
        if k.startswith("class_head.class_head"):
            new_k = k.replace("class_head.class_head", "class_head")
        elif k.startswith("bbox_head.bbox_head"):
            new_k = k.replace("bbox_head.bbox_head", "bbox_head")
        elif k.startswith("landmark_head.landmark_head"):
            new_k = k.replace("landmark_head.landmark_head", "landmark_head")
        
        adapted_state[new_k] = v

    # Perform fusion
    for base_name, params in fusion_buffer.items():
        conv_b = params.get("conv_bias")
        bn_b = params.get("bn_bias")
        fused_w, fused_b = fuse_bn_weights(
            params["conv_weight"], conv_b,
            params["bn_running_mean"], params["bn_running_var"],
            params["bn_weight"], bn_b
        )
        adapted_state[f"{base_name}.weight"] = fused_w
        adapted_state[f"{base_name}.bias"] = fused_b

    return adapted_state


def load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    state = torch.load(weights_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        state = adapt_state_dict(state)

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys when loading weights: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys when loading weights: {incompatible.unexpected_keys}")


def export_to_onnx(model: torch.nn.Module, onnx_path: Path, image_size: int, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size)
    dynamic_axes = {
        "input": {0: "batch"},
        "loc": {0: "batch"},
        "conf": {0: "batch"},
        "landms": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["loc", "conf", "landms"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"ONNX export completed: {onnx_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RetinaFace MobileNet0.25 to ONNX.")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("./../DockerOnnxDetector/retinaface_mv1_0.25.pth"),
        help="Path to the RetinaFace MobileNet0.25 checkpoint.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("artifacts/retinaface_mnet025.onnx"),
        help="Destination path for the exported ONNX file.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Export resolution (square). Adjust to match your inference input size.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=9,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = download_weights(args.weights_path)

    model = RetinaFace(phase="test", width_mult=0.25)
    load_weights(model, weights_path)
    model.eval()

    export_to_onnx(model, args.onnx_path, args.image_size, args.opset)


if __name__ == "__main__":
    main()
