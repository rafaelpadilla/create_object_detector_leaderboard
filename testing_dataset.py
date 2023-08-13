# Example: 
# python testing_dataset.py --path_models_csv ~/my_files/object_detection_models-08-08-2023.csv --output_folder ~/my_files/results

from fire import Fire
import evaluate
from datasets import load_dataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing
from pathlib import Path
import pandas as pd
import ast 
import json
from cocodataset import create_json_COCO_format
from cocodataset import COCODataset

from src.utils.basics import draw_and_save_boxes, get_current_date_time, map_func
from src.hf_models import load_model_predictor_from_name
MODELS_TO_SKIP = ["nielsr/deta-swin-large", "nielsr/deta-resnet-50", 
                  "nielsr/detr-resnet-50", "deepnight-research/object-detection-small", 
                  "hf-internal-testing/tiny-random-yolos", "nielsr/detr-resnet-50-new", 
                  "api19750904/cupra", "connorhoehn/detr_trading_card_display_detection_v1", 
                  "Amogh06/detr-for-table-detection", "Amogh06/detr-for-table-detection-v3", 
                  "nielsr/detr-finetuned-boat-detection", "SenseTime/deformable-detr-with-box-refine-two-stage"]

COCO_IGNORED_CLASSES = [
        "None",
        "street sign",
        "hat",
        "shoe",
        "eye glasses",
        "plate",
        "mirror",
        "window",
        "desk",
        "door",
        "blender",
        "hair brush",    
    ]

def benchmark_model(model_id, device, loaded_json_gts, coco_gt, batch_size):
    
    model_predictor = load_model_predictor_from_name(model_id, device=device, return_batch=True)
    
    ds_id2label = {cat["id"]: cat["name"] for cat in loaded_json_gts["categories"]} 
    ds_label2id = {cat["name"]: cat["id"] for cat in loaded_json_gts["categories"]} 
    if model_predictor is None:
        print(f"Error while initializing model {model_id}")
        return {}
    
    print(f"\nEvaluation of model {model_id} started:")
    
    # Clean classes
    evaluator = evaluate.load("rafaelpadilla/detection_metrics", json_gt=loaded_json_gts, iou_type="bbox")
    # Clean classes

    # Prepare dataloader
    num_workers = multiprocessing.cpu_count()
    val_dataloader = DataLoader(coco_gt, batch_size=batch_size, 
                                num_workers=num_workers, 
                                collate_fn=model_predictor.collate_fn)

    # Loop over the dataset
    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="Evaluating batches")
        for idx, batch in enumerate(pbar):
            
            predictions = model_predictor.predict_boxes(batch)
            
            # Fix labels mappings
            for pred in predictions:
                pred["labels"] = [ds_label2id[map_func(label_name)] for label_name in pred["label_names"]]

            # Contains a list of dictionaries having "image_id" as a list
            image_ids = [{"image_id": ref["image_id"]} for ref in batch["references"]]
            
            # Uncomment the lines below to save images with boxes
            # dir_to_save_images = 'images_with_boxes'
            # prefix = f"image_batch_{idx}"
            # draw_and_save_boxes(predictions, batch["raw_batch"], ds_id2label, prefix, dir_to_save_images)

            evaluator.add(prediction=predictions, reference=image_ids)
            
            del batch

    print(f"Results model: {model_id}")
    results = evaluator.compute()
    return results

def are_classes_comparable(lst_target_classes, dict_output_classes):
    keys = list(dict_output_classes.keys())
    values = list(dict_output_classes.values())
    
    reduce = lambda lst: [el for el in lst if el not in COCO_IGNORED_CLASSES]
    reduced_lst_target_classes = reduce(lst_target_classes)
    reduced_keys = reduce(keys)
    reduced_values = reduce(values)
    
    # Combine versions of reduced target classes and output_classes
    pairs = [
        (lst_target_classes, keys), \
        (lst_target_classes, values), \
        (lst_target_classes, reduced_keys), \
        (lst_target_classes, reduced_values), \
        (reduced_lst_target_classes, keys), \
        (reduced_lst_target_classes, values), \
        (reduced_lst_target_classes, reduced_keys), \
        (reduced_lst_target_classes, reduced_values) 
        ]
    for g1, g2 in pairs:
        if set(g1) == set(g2):
            return True
    return False

def main(path_models_csv: Path, output_folder: Path, dataset_name: str = "rafaelpadilla/coco2017", batch_size = 1):
    
    device = "cuda:0"

    # Get benchmarking dataset
    benchmark_dataset = load_dataset(dataset_name, split="val")
    target_classes = benchmark_dataset.features["objects"].feature["label"].names
    
    # Read models from csv
    df = pd.read_csv(path_models_csv) 
    
    models_to_benchmark = []
    # Get the models which have the target object classes
    for _, row in df.iterrows():
        
        output_classes_dict = ast.literal_eval(row["output_classes"])
        
        if are_classes_comparable(target_classes, output_classes_dict):
            models_to_benchmark.append(row["id"])
    
    # Create a JSON in memory that represents the COCO dataset
    ids_mapping, loaded_json_gts = create_json_COCO_format(benchmark_dataset, round_approx=2)
    
    # Create ground-truth dataset based on the JSON
    coco_gt = COCODataset(loaded_json_gts, ids_mapping, benchmark_dataset)

    output_folder.mkdir(parents=True, exist_ok=True)
    
    for model_id in models_to_benchmark:
        
        # Skipping models that require further investigation
        if model_id in MODELS_TO_SKIP:
            continue
        
        # Get info to save
        date_time = get_current_date_time()
        str_model_id = model_id.replace("/","@")
        fp = output_folder / f"{str_model_id}@@{date_time}.json"
        
        # If file with this name exists, skip
        matching_files = list(output_folder.glob(str_model_id+"*.json"))
        if len(matching_files) > 0:
            continue
        
        try:
            # Run benchmark
            metrics = benchmark_model(model_id = model_id,
                                    device=device, 
                                    loaded_json_gts=loaded_json_gts, 
                                    coco_gt=coco_gt,
                                    batch_size=batch_size)
            # Save
            with open(fp, "w") as file:
                json.dump(metrics, file, indent=4)
        except Exception as e:
            print(f"{e}: Error model {model_id}")

if __name__ == '__main__':
    Fire(main)  
