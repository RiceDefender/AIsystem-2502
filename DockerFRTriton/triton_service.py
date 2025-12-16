import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, List, Union

import numpy as np

TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002
MODEL_NAME = "fr_model"
MODEL_VERSION = "1"
MODEL_INPUT_NAME = "input"
MODEL_OUTPUT_NAME = "embedding"
MODEL_IMAGE_SIZE = (640, 640)


def prepare_model_repository(model_repo: Path) -> None:
    """
    Populate the Triton model repository with the FR ONNX model and config.pbtxt.
    """
    model_name = "fr_model"
    model_version = "1"
    model_dir = model_repo / model_name / model_version
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
        max_batch_size: 0
        default_model_filename: "model.onnx"
        input [
          {{
            name: "input"
            data_type: TYPE_FP32
            dims: [ 1, 3, 112, 112 ] 
          }}
        ]

        output [
          {{
            name: "embedding"
            data_type: TYPE_FP32
            dims: [ 1, 512 ]
          }}
        ]
        instance_group [
          {{ kind: KIND_CPU }}
        ]
        """
    ).strip() + "\n"

    config_path.write_text(config_text)
    print(f"[triton] Prepared model repository for {model_name} at {model_dir.parent}")


def prepare_face_detector_model_repository(model_repo: Path) -> None:
    """
    Populate the Triton model repository with the Face Detector ONNX model and config.pbtxt.
    """
    model_name = "face_detector"
    model_version = "1"
    model_dir = model_repo / model_name / model_version
    model_path = model_dir / "model.onnx"
    config_path = model_dir.parent / "config.pbtxt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing ONNX model at {model_path}. "
            "Run export_retinaface_to_onnx.py first or place your exported model there."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    # This config is for a RetinaFace-like detector.
    config_text = textwrap.dedent(
        f"""
        name: "{model_name}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        default_model_filename: "model.onnx"
        input [
          {{
            name: "input"
            data_type: TYPE_FP32
            dims: [ 3, 640, 640 ]
          }}
        ]
        output [
          {{
            name: "cat_3"
            data_type: TYPE_FP32
            dims: [ -1, 4 ]
            label_filename: "bboxes"
          }},
          {{
            name: "softmax"
            data_type: TYPE_FP32
            dims: [-1, 2]
            label_filename: "scores"
          }},
          {{
            name: "cat_5"
            data_type: TYPE_FP32
            dims: [-1, 10]
            label_filename: "landmarks"
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
    time.sleep(5) # Give the server a moment to load the model
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
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] is required") from exc

    client = httpclient.InferenceServerClient(url=url, verbose=False)
    return client


def run_inference(
        client: Any,
        image_bytes: bytes,
        model_name: str = "fr_model",
        input_name: str = "input",
        output_names: Union[str, List[str]] = "embedding",
        model_image_size: tuple[int, int] = (112, 112),
) -> Any:
    """
    Preprocess an input image, call Triton, and decode embeddings or scores.
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("triton http is required") from exc

    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize(model_image_size)
        np_img = np.asarray(img, dtype=np.float32)
        np_img = (np_img - 127.5) / 128.0

    np_img = np.transpose(np_img, (2, 0, 1))
    batch = np.expand_dims(np_img, axis=0)

    infer_input = httpclient.InferInput(input_name, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    if isinstance(output_names, str):
        output_names = [output_names]

    outputs_to_request = [httpclient.InferRequestedOutput(name) for name in output_names]

    response = client.infer(model_name=model_name, inputs=[infer_input], outputs=outputs_to_request)

    if len(output_names) == 1:
        return response.as_numpy(output_names[0])

    return {name: response.as_numpy(name) for name in output_names}