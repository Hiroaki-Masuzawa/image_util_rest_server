from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import torch
from PIL import Image
from io import BytesIO
from typing import Dict
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

app = FastAPI()

# --- モデル登録処理 ---

def load_maskrcnn_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    return {"predictor": predictor, "metadata": metadata}

# モデル名とその処理インスタンスのマップ
model_registry: Dict[str, Dict] = {
    "maskrcnn": load_maskrcnn_model()
}

# --- 画像予測処理 ---

@app.post("/{model_name}/predict")
async def predict(model_name: str, file: UploadFile = File(...)):
    if model_name not in model_registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    model_data = model_registry[model_name]
    predictor = model_data["predictor"]
    metadata = model_data["metadata"]

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    outputs = predictor(image_np)
    instances = outputs["instances"].to("cpu")

    results = []

    for i in range(len(instances)):
        box = instances.pred_boxes.tensor[i].numpy().tolist()
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        confidence = float(instances.scores[i])
        class_id = int(instances.pred_classes[i])
        class_name = metadata.get("thing_classes", [])[class_id] if metadata.get("thing_classes") else str(class_id)

        mask = instances.pred_masks[i].numpy().astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                points.append({"x": int(x), "y": int(y)})

        results.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "x": center_x,
            "y": center_y,
            "width": width,
            "height": height,
            "mask": mask.tolist(),
            "points": points,
        })

    return JSONResponse(content={"predictions": results})


@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return {"available_models": list(model_registry.keys())}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)
