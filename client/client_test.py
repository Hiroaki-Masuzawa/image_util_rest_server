import requests


def get_instance_segmentation(
    image_path,
    servername="localhost",
    port=8000,
    model="ram-grounded-sam",
    prompt="",
    debug=False,
):
    if image_path.split(".")[-1] == "jpg":
        imagetype = "png"
    else:
        imagetype = "jpeg"

    url = "http://{}:{}/{}/predict".format(servername, port, model)
    data = {}
    if prompt != "":
        data["text_prompt"] = prompt
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/{}".format(imagetype))}
        response = requests.post(url, files=files, data=data)

    response_dict = response.json()
    if debug:
        debug_image = make_debug_image(image_path, response_dict)
        cv2.imwrite("output.png", debug_image)
    return response_dict


def make_debug_image(image_path, result):
    image = cv2.imread(image_path)
    colors = (
        (0, 0, 255),
        # (0, 128, 255),
        (0, 255, 255),
        # (0, 255, 128),
        (0, 255, 0),
        # (128, 255, 0),
        (255, 255, 0),
        # (255, 128, 0),
        (255, 0, 0),
        # (255, 0, 128),
        (255, 0, 255),
        # (128, 0, 255),
    )
    for i in range(len(result["predictions"])):
        xs, ys, xe, ye = np.array(result["predictions"][i]["bbox"]).astype(np.int64)
        color = colors[i % len(colors)]
        cv2.rectangle(image, (xs, ys), (xe, ye), color, 3)
        mask = np.array(result["predictions"][i]["mask"]).astype(np.int64)
        for v in range(mask.shape[0]):
            for u in range(mask.shape[1]):
                if mask[v, u] != 0:
                    image[v, u] = (image[v, u] + np.array(color)) // 2
    return image


import cv2
import numpy as np
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
