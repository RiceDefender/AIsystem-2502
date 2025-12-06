import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np


TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002
MODEL_NAME = "fr_model"
MODEL_VERSION = "1"
MODEL_INPUT_NAME = "input"
MODEL_OUTPUT_NAME = "embedding"
MODEL_IMAGE_SIZE = (112, 112)


def prepare_model_repository(model_repo: Path) -> None:
    """
    Populate the Triton model repository with the FR ONNX model and config.pbtxt.
    """
    model_dir = model_repo / MODEL_NAME / MODEL_VERSION
    model_path = model_dir / "model.onnx"
    config_path = model_dir.parent / "config.pbtxt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing ONNX model at {model_path}. "
            "Run convert_to_onnx.py first or place your exported model there."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    config_text = textwrap.dedent(
        f"""
        name: "{MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        default_model_filename: "model.onnx"
        input [
          {{
            name: "{MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [3, {MODEL_IMAGE_SIZE[0]}, {MODEL_IMAGE_SIZE[1]}]
          }}
        ]
        output [
          {{
            name: "{MODEL_OUTPUT_NAME}"
            data_type: TYPE_FP32
            dims: [512]
          }}
        ]
        instance_group [
          {{ kind: KIND_CPU }}
        ]
        """
    ).strip() + "\n"

    config_path.write_text(config_text)
    print(f"[triton] Prepared model repository at {model_dir.parent}")


def start_triton_server(model_repo: Path) -> Any:
    """
    Launch Triton Inference Server (CPU) pointing at model_repo and return a handle/process.
    """
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH. Is Triton installed?")

    cmd = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--allow-metrics=true",
        "--log-verbose=1",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[triton] Starting Triton server with command: {' '.join(cmd)}")
    time.sleep(3)  # Give the server a moment to load the model
    return process


def stop_triton_server(server_handle: Any) -> None:
    """
    Cleanly stop the Triton server started in start_triton_server.
    """
    if server_handle is None:
        return

    server_handle.terminate()
    try:
        server_handle.wait(timeout=10)
        print("[triton] Triton server stopped.")
    except subprocess.TimeoutExpired:
        server_handle.kill()
        print("[triton] Triton server killed after timeout.")


def create_triton_client(url: str) -> Any:
    """
    Initialize a Triton HTTP client for the FR model endpoint.
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("tritonclient[http] is required; install from requirements.txt") from exc

    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return client


def run_inference(client: Any, image_bytes: bytes) -> Any:
    """
    Preprocess an input image, call Triton, and decode embeddings or scores.
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Pillow, numpy, and tritonclient[http] are required to run inference.") from exc

    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize(MODEL_IMAGE_SIZE)
        np_img = np.asarray(img, dtype=np.float32) / 255.0

    np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
    batch = np.expand_dims(np_img, axis=0)

    infer_input = httpclient.InferInput(MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    infer_output = httpclient.InferRequestedOutput(MODEL_OUTPUT_NAME)
    response = client.infer(model_name=MODEL_NAME, inputs=[infer_input], outputs=[infer_output])
    return response.as_numpy(MODEL_OUTPUT_NAME)
