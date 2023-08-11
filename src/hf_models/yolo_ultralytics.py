from .base import ModelEvaluator
from src.utils.basics import convert_to_xywh
from transformers import YolosFeatureExtractor

class YoloUltralytics(ModelEvaluator):
    """ Class for evaluating object detection with Detr."""
    
    def __init__(self, model_obj, model_id: str, device: str, return_batch: bool = False, threshold: float = 0.):
        # By default YOLO's confidence is set to 0.25. For AP evaluation, we need to set it to 0.
        model_obj.conf = threshold
        # Yolo v5 does not have classes as model.config
        id2label = model_obj.names
        im_processor_initiator = YolosFeatureExtractor
        super().__init__(model_obj, model_id, device, im_processor_initiator, return_batch, threshold, id2label=id2label)
    
    def _collate_fn(self, batch):
        images = [sample["image"] for sample in batch]
        orig_sizes = [sample["image"].size[::-1] for sample in batch]  # h, w
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
        return {"images": images, "orig_sizes": orig_sizes, "references": references}
    
    def predict_boxes(self, batch):
        # Transform into tensors
        predictions = []
        outputs = self.model.predict(batch["images"])
        for pred in outputs:
            # pred.boxes.xywh does not work, as it returns based on the resized image.
            # thus, we obtain xyx2y2 and convert to xywh
            xywh = convert_to_xywh(pred.boxes.xyxy)
            names = [pred.names[int(class_id)] for class_id in pred.boxes.cls]
            predictions.append({"scores": pred.boxes.conf, 
                        "labels": pred.boxes.cls, 
                        "boxes": xywh, 
                        "label_names": names})
        return predictions