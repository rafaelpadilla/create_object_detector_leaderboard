from .base import ModelEvaluator
from src.utils.basics import convert_to_xywh

class Yolov5(ModelEvaluator):
    """ Class for evaluating object detection with Yolo."""
    
    def __init__(self, model_obj, model_id: str, device: str, return_batch: bool = False, threshold: float = 0.):
        # By default YOLO's confidence is set to 0.25. For AP evaluation, we need to set it to 0.
        model_obj.conf = threshold
        model_obj.iou = 0.
        # Yolo v5 does not have classes as model.config
        id2label = model_obj.names
        super().__init__(model_obj, model_id, device, return_batch = return_batch, threshold = threshold, id2label=id2label)
    
    def _collate_fn(self, batch):
        images = [sample["image"] for sample in batch]
        # Get some stuff for the ground-truths
        references = []
        for sample in batch:
            image_id = sample["target"]["image_id"]
            sample_ref = {"image_id": [image_id], "bbox": [], "category_id": [], "iscrowd":[]}
            for annot in sample["target"]["annotations"]:
                sample_ref["bbox"].append(annot["bbox"])
                sample_ref["category_id"].append(annot["category_id"])
                sample_ref["iscrowd"].append(annot["iscrowd"])
            references.append(sample_ref)
        return {"images": images, "references": references}

    def predict_boxes(self, batch):
        # forward pass
        outputs = self.model(batch["images"])
        ret = []
        for prediction in outputs.pred:
            boxes = prediction[:, :4] # x1, y1, x2, y2
            boxes = convert_to_xywh(boxes) # x, y, w, h
            scores = prediction[:, 4]
            category_ids = prediction[:, 5]
            category_names = [self.id2label[l.item()] for l in category_ids]
            ret.append({"scores": scores, "labels": category_ids, "boxes": boxes, "label_names": category_names})
        return ret
        