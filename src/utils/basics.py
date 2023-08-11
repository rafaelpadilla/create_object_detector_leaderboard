from datetime import datetime
import pytz
import torch
from pathlib import Path
from cocodataset import draw_rectangles

def get_current_date_time() -> str:
    """
    Get the current date and time in the GMT timezone in the format 'MM-DD-YYYY_HH_MM_SS'.

    Returns:
        str: The current date and time in the GMT timezone in the format 'MM-DD-YYYY_HH_MM_SS'.
    """
    gmt = pytz.timezone('GMT')
    current_date = datetime.now(gmt)
    formatted_date = current_date.strftime("%b-%d-%Y_%Hh%Mm%Ss")
    return formatted_date

def _invert_mapping(mapping):
    ret = {}
    for k,v in mapping.items():
        for x in v:
            ret[x] = k
    return ret

def map_func(x):
    mapping = {"None": {"None", "N/A"}}
    mapping = _invert_mapping(mapping)
    return mapping[x] if x in mapping else x

# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def xywhn2xywh(size, box):
    # w_box = round(size[0] * box[2])
    # h_box = round(size[1] * box[3])
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    # return (xIn, yIn, xEnd, yEnd)
    return (xIn, yIn, xEnd-xIn, yEnd-yIn)

def convert_to_xywh(boxes):
    # From xyx2y2 to xywh
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def convert_to_xyx2y2(boxes):
    # From xywh to xyx2y2
    xmin, ymin, w, h = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmin + w, ymin + h), dim=1)

def draw_and_save_boxes(predictions, raw_batch, id2label, prefix, dir_to_save):
    if isinstance(dir_to_save, str):
        dir_to_save = Path(dir_to_save)
    dir_to_save.mkdir(parents=True, exist_ok=True)
    for sample_id, sample in enumerate(raw_batch):
        image = sample["image"]
        gt_boxes = [annot["bbox"] for annot in sample["target"]["annotations"]]
        gt_label_ids = [annot["category_id"] for annot in sample["target"]["annotations"]]
        gt_label_cats = [id2label[idx] for idx in gt_label_ids]
        # Draw gts in image
        img_bbx = draw_rectangles(image=image, boxes=gt_boxes, box_format="xywh", labels=gt_label_cats, color_bbx=(0,255,0))
        # Draw predictions in image
        boxes = predictions[sample_id]["boxes"]
        labels = predictions[sample_id]["label_names"]
        confidences = predictions[sample_id]["scores"]
        img_bbx = draw_rectangles(image=img_bbx, boxes=boxes, labels=labels, confidences=confidences, box_format="xywh", color_bbx=(255,0,0))
        fp = dir_to_save / f"{prefix}_{sample_id}.png"
        img_bbx.save(fp)
