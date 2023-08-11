from .base import ModelEvaluator
from src.utils.basics import convert_to_xywh, xywhn2xywh
from transformers import YolosImageProcessor

class Yolo(ModelEvaluator):
    """ Class for evaluating object detection with Detr."""
    
    def __init__(self, model_obj, model_id: str, device: str, return_batch: bool = False, threshold: float = 0.):
        # By default YOLO's confidence is set to 0.25. For AP evaluation, we need to set it to 0.
        model_obj.conf = threshold
        # Yolo v5 does not have classes as model.config
        im_processor_initiator = YolosImageProcessor.from_pretrained
        super().__init__(model_obj, model_id, device, im_processor_initiator, return_batch, threshold)
    
    def _collate_fn(self, batch):
        images = [sample["image"] for sample in batch]
        orig_sizes = [sample["image"].size[::-1] for sample in batch]  # h, w
        encoding = self.im_processor(images=images, return_tensors="pt")
        pixel_values = encoding["pixel_values"]
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
        return {"pixel_values": pixel_values, "images": images, "orig_sizes": orig_sizes, "references": references}
    
    def predict_boxes(self, batch):
        # Get inputs
        pixel_values = batch["pixel_values"].to(self.device)
        orig_sizes = batch["orig_sizes"]

        outputs = self.model(pixel_values)
        results = self.im_processor.post_process_object_detection(outputs, target_sizes=orig_sizes, threshold=self.threshold)

        predictions = []
        for res in results:
            prediction = {
                "scores": res["scores"],
                "labels": res["labels"],
                "boxes": convert_to_xywh(res["boxes"]), 
                "label_names": [self.id2label[l.item()] for l in res["labels"]],
            }
            predictions.append(prediction)
        return predictions