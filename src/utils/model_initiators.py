from transformers import  AutoModelForObjectDetection, DetrForObjectDetection, OwlViTForObjectDetection, DeformableDetrForObjectDetection, AutoImageProcessor, DetrImageProcessor, ConditionalDetrForObjectDetection, DetaForObjectDetection, YolosForObjectDetection, DeformableDetrForObjectDetection
from ultralyticsplus import YOLO
import yolov5
import yolov7

models_initiators = [AutoModelForObjectDetection, DeformableDetrForObjectDetection, DetrForObjectDetection, OwlViTForObjectDetection, YOLO, yolov5, yolov7, YolosForObjectDetection, ConditionalDetrForObjectDetection, DetaForObjectDetection]
# image_processors = [AutoImageProcessor, DetrImageProcessor]

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

def try_create_processor(img_processor, model_id):
    try:
        return img_processor.from_pretrained(model_id)
    except:
        pass

def get_full_class_name(obj):
    klass = obj.__class__
    module = klass.__module__
    # To avoid output 'builtins.str'
    if module == 'builtins':
        return klass.__qualname__ 
    return module + '.' + klass.__qualname__

    
def initiate_model(model_id):
    for model_initiator in models_initiators:
        model = try_create_model(model_initiator, model_id)
        if model is not None:
            return model
    return None

# def initiate_processor(model_id):
#     for img_processor in image_processors:
#         processor = try_create_processor(img_processor, model_id)   
#         if processor is not None:
#             return processor
#     return None

# def infer_architecture(model):
#     if hasattr(model, "config") and hasattr(model.config, "architectures"):
#         return model.config.architectures
#     else:
#         return []
