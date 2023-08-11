from .base import ModelEvaluator
from src.utils.basics import convert_to_xywh
from transformers import AutoImageProcessor

class Deta(ModelEvaluator):
    """ Class for evaluating object detection with Deta."""
    
    def __init__(self, model_obj, model_id: str, device: str, return_batch: bool = False, threshold: float = 0.):
        im_processor_initiator = AutoImageProcessor.from_pretrained
        super().__init__(model_obj, model_id, device, im_processor_initiator, return_batch, threshold)
    
    def _collate_fn(self, batch):
        images = [sample["image"] for sample in batch]
        orig_sizes = [sample["image"].size[::-1] for sample in batch]  # h, w
        # Detr: sample["target"] must be a dict containing 
        #           "image_id" (str) and "annotations" (List[Dict])
        # target = [sample["target"] for sample in batch]
        encoding = self.im_processor(images=images, return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        pixel_mask = encoding["pixel_mask"]
        
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
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "orig_sizes": orig_sizes, "images": images, "references": references}

    
    def predict_boxes(self, batch):
        # Transform into tensors
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        orig_sizes = batch["orig_sizes"] # h, w
        # forward pass
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        # Provide target_size to obtain boxes in absolute format
        predictions = self.im_processor.post_process_object_detection(outputs=outputs, target_sizes=orig_sizes, threshold=self.threshold) 
        for idx, pred_batch in enumerate(predictions):
            # Originally boxes boxes are in xyx2y2 format. COCO evaluator requires xywh
            boxes_xywh = convert_to_xywh(pred_batch["boxes"])
            pred_batch["boxes"] = boxes_xywh
            pred_batch["label_names"] = [self.id2label[l.item()] for l in pred_batch["labels"]]
        return predictions