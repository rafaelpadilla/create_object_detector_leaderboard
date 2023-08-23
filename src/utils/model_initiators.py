from transformers import  AutoModelForObjectDetection, DetrForObjectDetection, OwlViTForObjectDetection, DeformableDetrForObjectDetection, AutoImageProcessor, DetrImageProcessor, ConditionalDetrForObjectDetection, DetaForObjectDetection, YolosForObjectDetection, DeformableDetrForObjectDetection
from ultralyticsplus import YOLO
import yolov5
import yolov7
from ..hf_models.detr import Detr
from ..hf_models.owl import Owl
from ..hf_models.deformable_detr import DeformableDetr
from ..hf_models.deta import Deta
from ..hf_models.yolo import Yolo
from ..hf_models.yolo_ultralytics import YoloUltralytics
from ..hf_models.yolov5 import Yolov5
from ..hf_models.conditional_detr import ConditionalDetr

# Maps model's class name to a class in hf_models
dict_architecture = {"transformers.models.detr.modeling_detr.DetrForObjectDetection": Detr,
                     "transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection": Owl,
                     "yolov5.models.common.AutoShape": Yolov5,
                     "ultralyticsplus.ultralytics_utils.YOLO": YoloUltralytics, 
                     "transformers.models.yolos.modeling_yolos.YolosForObjectDetection": Yolo,
                     "transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrForObjectDetection": ConditionalDetr,
                     "transformers.models.deta.modeling_deta.DetaForObjectDetection": Deta,
                     "transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrForObjectDetection": DeformableDetr
                     }

dict_model_initiator = {"owlvit-base-patch32": OwlViTForObjectDetection,
                        "owlvit-large-patch14": OwlViTForObjectDetection,
                        "owlvit-base-patch16": OwlViTForObjectDetection,}

models_initiators = [AutoModelForObjectDetection, DeformableDetrForObjectDetection, DetrForObjectDetection, OwlViTForObjectDetection, YOLO, yolov5, yolov7, YolosForObjectDetection, ConditionalDetrForObjectDetection, DetaForObjectDetection]

def get_classes_from_model(model):
    if hasattr(model, "config") and hasattr(model.config, "label2id"):
        return model.config.label2id
    elif hasattr(model, "names"):
        return model.names
    else:
        return {}

def try_create_model(model_class, model_id):
    try:
        return model_class.from_pretrained(model_id)
    except:
        pass
    try:
        return model_class.load(model_id)
    except:
        pass
    try:
        return model_class(model_id)
    except:
        pass
    return None


def get_full_class_name(obj):
    klass = obj.__class__
    module = klass.__module__
    # To avoid output 'builtins.str'
    if module == 'builtins':
        return klass.__qualname__ 
    return module + '.' + klass.__qualname__

def _split_org_model(model_id: str) -> str:
    sep = "/"
    if sep in model_id:
        return model_id.split(sep)
    return model_id, ""
    
def initiate_model(model_id):
    # First check if model is defined in dict_model_initiator
    _, model_name = _split_org_model(model_id)
    _class = dict_model_initiator.get(model_name)
    
    lst_model_initiators = models_initiators
    
    # If model is not mapped in dict_model_initiator
    if _class is not None:
        lst_model_initiators = [_class]
    
    for model_initiator in lst_model_initiators:
        model = try_create_model(model_initiator, model_id)
        if model is not None:
            return model
    return None


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

