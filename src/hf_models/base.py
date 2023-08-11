from typing import Callable, List, Optional
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
from typing import Dict

class ModelEvaluator(object):
    
    def __init__(self, model_obj, model_id: str, device: str, im_processor_initiator: Optional[Callable] = None, return_batch: bool = False, threshold: float = 0., id2label=None):
        self.model_id = model_id
        self.device = device
        self.return_batch = return_batch
        self.threshold = threshold
        self.model = model_obj
        # Yolo's .to(device) returns None
        _res = self.model.to(device)
        if _res is not None:
            self.model = _res
        # Converter id2label
        if id2label is None:
            self.id2label = self.model.config.id2label
        else:
            self.id2label = id2label
        # Converter label2id
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        if im_processor_initiator is not None:
            self.im_processor = im_processor_initiator(model_id, do_normalize_annotations=False)
        else:
            self.im_processor = None
    
    def collate_fn(self, batch):
        ret = self._collate_fn(batch)
        if self.return_batch:
            ret["raw_batch"] = batch
        return ret
    
    @property
    def sorted_classes(self) -> List[str]:
        classes_model = sorted(self.model.config.label2id.items(), key=lambda x: x[1])
        classes_model = [c[0] for c in classes_model]
        return classes_model

    @property
    def total_classes(self) -> int:
        return len(self.sorted_classes)
        
    def _collate_fn(self, batch):
        raise NotImplementedError(f"Method not implemented for {self.model_id}")
    
    def predict_boxes(self, batch) -> Dict:
        raise NotImplementedError(f"Method not implemented for {self.model_id}")