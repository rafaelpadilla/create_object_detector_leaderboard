from .base import ModelEvaluator
from src.utils.basics import convert_to_xywh
from transformers import OwlViTProcessor
import torch

class Owl(ModelEvaluator):
    """ Class for evaluating object detection with OwlVit."""
    
    def __init__(self, model_obj, model_id: str, device: str, return_batch: bool = False, threshold: float = 0.):
        im_processor_initiator = OwlViTProcessor.from_pretrained
        self._target_classes = {}
        super().__init__(model_obj, model_id, device, im_processor_initiator, return_batch, threshold)
    
    @property
    def target_classes(self):
        return self._target_classes
    
    @target_classes.setter
    def target_classes(self, target_classes):
        self._target_classes = {}
        self.query_texts = {}
        for idx, c in target_classes.items():
            if c.lower() in ["none"]:
                self._target_classes[idx] = None
            else:
                self._target_classes[f"an image of a {c}"] = idx
                self.query_texts[len(self.query_texts)] = f"an image of a {c}"
        
    def _collate_fn(self, batch):
        images = [sample["image"] for sample in batch]
        orig_sizes = [sample["image"].size[::-1] for sample in batch]  # h, w
        text = list(self.query_texts.values())
        text = [text for i in range(len(images))]
        encoding = self.im_processor(text=text, images=images, return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
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
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "orig_sizes": orig_sizes, "images": images, "references": references}

    
    def predict_boxes(self, batch):
        # Transform into tensors
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        orig_sizes = torch.Tensor(batch["orig_sizes"]).to(self.device) # h, w
        # forward pass
        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        # Provide target_size to obtain boxes in absolute format
        predictions = self.im_processor.post_process_object_detection(outputs=outputs, target_sizes=orig_sizes, threshold=self.threshold) 
        for idx, pred_batch in enumerate(predictions):
            pred_batch["scores"] = pred_batch["scores"]
            pred_batch["labels"] = pred_batch["labels"]
            # Originally boxes boxes are in xyx2y2 format. COCO evaluator requires xywh
            boxes_xywh = convert_to_xywh(pred_batch["boxes"])
            pred_batch["boxes"] = boxes_xywh
            # Get the text prompts based on the labels
            txt_prompts = [self.query_texts[i.item()]  for i in pred_batch["labels"]]
            # Match the labels prompts with the target_classes (database)
            pred_batch["labels"] = [self.target_classes[txt] for txt in txt_prompts]
            # Get only label names (useful for drawing bounding boxes)
            pred_batch["label_names"] = [txt.replace("an image of a ","") for txt in txt_prompts]
        return predictions