from .detr import Detr
from .deformable_detr import DeformableDetr
from .deta import Deta
from.yolo import Yolo
from .yolo_ultralytics import YoloUltralytics
from .yolov5 import Yolov5
from .conditional_detr import ConditionalDetr
from src.utils.model_initiators import initiate_model, get_full_class_name

dict_architecture = {"transformers.models.detr.modeling_detr.DetrForObjectDetection": Detr,
                     "yolov5.models.common.AutoShape": Yolov5,
                     "ultralyticsplus.ultralytics_utils.YOLO": YoloUltralytics, 
                     "transformers.models.yolos.modeling_yolos.YolosForObjectDetection": Yolo,
                     "transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrForObjectDetection": ConditionalDetr,
                     "transformers.models.deta.modeling_deta.DetaForObjectDetection": Deta,
                     "transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrForObjectDetection": DeformableDetr
                     }

def load_model_predictor_from_name(model_id: str, device: str = "cuda", return_batch: bool = False, threshold: float = 0.):
    # Try to load a model
    model = initiate_model(model_id)
    
    if model is None:    
        # raise ValueError(f"Unable to load model {model_id}")
        print(f"Unable to load model {model_id}")
        return None
    
    class_name = get_full_class_name(model)
    _class = dict_architecture[class_name]
    return _class(model, model_id, device, return_batch, threshold)
