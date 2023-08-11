from typing import Dict
from src.utils.basics import map_func
from src.utils.model_initiators import initiate_model, get_classes_from_model
import huggingface_hub as hf_hub
import pandas as pd
from tqdm import tqdm
from src.utils.basics import get_current_date_time

_ACCEPTED_FILTERS = ["object-detection"]

def get_models_output_classes(model_id):
    # Get all classes output by the model
    output_classes = {}
    model = initiate_model(model_id)
    if model is None:
        print(f"Not able to initiate model {model_id}")
        return output_classes
    
    output_classes = get_classes_from_model(model)
    if len(output_classes) == 0:
        print(f"Not able to retrieve classes from model {model.__name__}")
        return output_classes
    
    output_classes = sorted(output_classes.items(), key=lambda x: x[1])
    # Make "None" and "N/A" equivalent
    output_classes = {c[1]: map_func(c[0]) for c in output_classes}
    return output_classes

def get_models_from_hub(filter="object-detection") -> pd.DataFrame:
    assert filter in _ACCEPTED_FILTERS, f"Filter must be one of {_ACCEPTED_FILTERS}"
    
    # Get only models sorted by likes (optional: sort by downloads)
    obj_det_models = hf_hub.list_models(filter=filter, sort="likes", direction=-1)
    obj_det_models = list(obj_det_models)

    dict_data = {"_id": [], "id": [], "author": [], "downloads": [], "lastModified": [], "library_name": [], "likes": [], "output_classes": []}
    pbar = tqdm(obj_det_models, total=len(obj_det_models))
    for model in pbar:
        pbar.set_description(f"Fetching model {model.id}")
        dict_data["_id"].append(model._id)
        dict_data["id"].append(model.id)
        dict_data["author"].append(model.author)
        dict_data["downloads"].append(model.downloads)
        dict_data["lastModified"].append(model.lastModified)
        library_name = model.library_name if hasattr(model, "library_name") else ""
        dict_data["library_name"].append(library_name)
        dict_data["likes"].append(model.likes)
        # Get classes output by the model sorted by id
        output_classes = get_models_output_classes(model.id)
        dict_data["output_classes"].append(output_classes)
    
    return pd.DataFrame.from_dict(dict_data)

def main(model_type: str = "object-detection"):
    # Fetch object detection pre-trained models from HuggingFace Hub
    df_models = get_models_from_hub(filter=model_type)
    total_models = len(df_models)
    print(f"{total_models} {model_type} models were found in the HuggingFace Hub.")
    
    # Sort by column "output_classes"
    df_models["output_classes"] = df_models["output_classes"].astype('string')
    df_models.sort_values(by=["output_classes"], inplace=True)

    # Save to CSV
    current_date_time = get_current_date_time()
    df_models.to_csv(f"object_detection_models-{current_date_time}.csv", index=False)

if __name__ == "__main__":
    main()
