from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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


from segment_anything import (
    build_sam,
    build_sam_vit_b,
    build_sam_hq,
    SamPredictor
) 
import GroundingDINO.groundingdino.datasets.transforms as T
from automatic_label_ram_demo import load_model
import torchvision.transforms as TS
from ram.models import ram
from ram import inference_ram
from automatic_label_ram_demo import get_grounding_output
import torchvision
import re

def load_RGS_model():
    None

def pred_RGS_model(image_pil, text_prompt):
    config_file = '/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    grounded_checkpoint = '/Grounded-Segment-Anything/groundingdino_swint_ogc.pth'
    ram_checkpoint = '/Grounded-Segment-Anything/ram_swin_large_14m.pth'
    sam_checkpoint_h = '/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
    sam_checkpoint_b = '/Grounded-Segment-Anything/sam_vit_b_01ec64.pth'
    box_threshold = 0.25
    text_threshold  = 0.2
    iou_threshold = 0.5
    device = 'cuda'

    transform1 = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image, _ = transform1(image_pil, None) 

    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform2 = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    if text_prompt is None:
        ram_model = ram(pretrained=ram_checkpoint,
                                            image_size=384,
                                            vit='swin_l')
        ram_model.eval()

        ram_model = ram_model.to(device)
        raw_image = image_pil.resize((384, 384))
        raw_image  = transform2(raw_image).unsqueeze(0).to(device)
        res = inference_ram(raw_image , ram_model)
        tags = res[0].replace(' |', ',')
    else :
        tags = text_prompt

    grounded_model = load_model(config_file, grounded_checkpoint, device=device)
    boxes_filt, scores, pred_phrases = get_grounding_output(
        grounded_model, image, tags, box_threshold, text_threshold, device=device
    )

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    boxes_filt = boxes_filt.cpu()
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]

    results = []
    if boxes_filt.shape[0] != 0:
        # predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint_h).to(device))
        predictor = SamPredictor(build_sam_vit_b(checkpoint=sam_checkpoint_b).to(device))
        image_np = np.array(image_pil) # RGB
        predictor.set_image(image_np)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        value  = 0
        mask_img = torch.zeros(masks.shape[-2:])
        for idx, mask in enumerate(masks):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

        for phrase, bbox, mask in zip(pred_phrases, boxes_filt.cpu(), masks.cpu()):
            match = re.match(r"(.+?)\s*\(([^()]*)\)", phrase)
            if match:
                class_name = match.group(1)
                confidence = float(match.group(2))
            else :
                class_name = phrase
                confidence = 1.0
            x1, y1, x2, y2 = bbox.numpy().astype(np.float64)
            center_x = (x1+x2)/2
            center_y = (y1+y2)/2
            width = (x2-x1)/2
            height = (y2-y1)/2

            contours, _ = cv2.findContours(mask[0].numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = []
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    points.append({"x": int(x), "y": int(y)})

            mask = mask.numpy().astype(np.int64)
            results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "x": float(center_x),
                    "y": float(center_y),
                    "width": float(width),
                    "height": float(height),
                    "mask": mask[0].tolist(),
                    "points": points,
                })
    return results


# モデル名とその処理インスタンスのマップ
model_registry: Dict[str, Dict] = {
    "maskrcnn": load_maskrcnn_model(), 
    "ram-grounded-sam": load_RGS_model()
}

# --- 画像予測処理 ---

@app.post("/{model_name}/predict")
async def predict(model_name: str,
                   file: UploadFile = File(...),
                   text_prompt: str = Form(None),
                   ):
    if model_name not in model_registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    if model_name == 'maskrcnn':
        model_data = model_registry[model_name]
        predictor = model_data["predictor"]
        metadata = model_data["metadata"]

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
    else:
        results = pred_RGS_model(image_pil=image, text_prompt=text_prompt)
        return JSONResponse(content={"predictions": results})

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return {"available_models": list(model_registry.keys())}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)
