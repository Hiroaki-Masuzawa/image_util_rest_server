import requests
from pathlib import Path
import cv2
import numpy as np


def get_instance_segmentation(
    image_path: str,
    servername: str = "localhost",
    port: int = 8000,
    model: str = "maskrcnn",
    prompt: str = "",
    debug: bool = False,
) -> dict:
    """
    Sends an image to an external instance segmentation server and returns the prediction results.

    The function posts the image to the specified server and model endpoint.
    Optionally, a text prompt can be included. If `debug` is True, it creates and saves
    a debug image with predicted bounding boxes and masks overlaid.

    Args:
        image_path (str): Path to the input image file.
        servername (str, optional): Hostname or IP address of the inference server. Defaults to "localhost".
        port (int, optional): Port number used by the server. Defaults to 8000.
        model (str, optional): Name of the model to use for prediction. Defaults to "maskrcnn".
        prompt (str, optional): Optional text prompt to include in the request. Defaults to "".
        debug (bool, optional): If True, saves a debug image with predictions overlaid. Defaults to False.

    Returns:
        dict: A dictionary containing the segmentation results, including bounding boxes and masks.

    Raises:
        RuntimeError: If the server returns a non-200 response or if the response is not valid JSON.
    """
    suffix = Path(image_path).suffix.lower()
    imagetype = "jpeg" if suffix in [".jpg", ".jpeg"] else "png"

    url = f"http://{servername}:{port}/{model}/predict"

    data = {"text_prompt": prompt} if prompt else {}

    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, f"image/{imagetype}")}
        response = requests.post(url, files=files, data=data)

    if response.status_code != 200:
        raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")

    try:
        response_dict = response.json()
    except ValueError:
        raise RuntimeError("Failed to parse JSON response from server")

    if debug:
        debug_image = _make_debug_image(image_path, response_dict)
        cv2.imwrite("output.png", debug_image)
    return response_dict



def _make_debug_image(image_path: str, result: dict) -> np.ndarray:
    image = cv2.imread(image_path)
    colors = [
        (0, 0, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (255, 0, 255),
    ]

    for i, pred in enumerate(result["predictions"]):
        xs, ys, xe, ye = np.array(pred["bbox"]).astype(np.int64)
        color = colors[i % len(colors)]
        cv2.rectangle(image, (xs, ys), (xe, ye), color, 3)

        mask = np.array(pred["mask"]).astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[:, :] = color
        image = np.where(mask[:, :, None] != 0, (image // 2 + colored_mask // 2), image)
        cv2.putText(image, "{0}:{1:.2f}".format(pred["class"], pred["confidence"]), (xs, ys), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))

    return image


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="maskrcnn",
        choices=["maskrcnn", "ram-grounded-sam"],
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    image_path = args.image_path
    result = get_instance_segmentation(
        image_path,
        servername=args.server,
        port=args.port,
        model=args.model,
        prompt=args.prompt,
        debug=args.debug,
    )

    for i in range(len(result["predictions"])):
        # for k in ['class', 'confidence', 'bbox', 'x', 'y', 'width', 'height']:
        for k in ["class", "confidence", "bbox"]:
            print(result["predictions"][i][k])
        print("-" * 32)
