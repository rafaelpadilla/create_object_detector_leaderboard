# Example: 
# python scripts/evaluate_models.py --path_models_csv ~/my_files/object_detection_models-May-26-2023.csv --output_folder ~/my_files/results
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
import time

from cocodataset import create_json_COCO_format
from cocodataset import COCODataset

from src.utils.basics import draw_and_save_boxes, get_current_date_time, map_func
from src.utils.model_initiators import load_model_predictor_from_name
from src.gpu_manager import MultiGPUManager


MODELS_TO_SKIP = ["deepnight-research/object-detection-small", 
                  "hf-internal-testing/tiny-random-yolos",  
                  "api19750904/cupra", "connorhoehn/detr_trading_card_display_detection_v1", 
                  "Amogh06/detr-for-table-detection", "Amogh06/detr-for-table-detection-v3", 
                  "SenseTime/deformable-detr-with-box-refine-two-stage"]

AUTHORS_TO_SKIP = ["nielsr", "amyeroberts"]

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

def benchmark_model(model_id, device, loaded_json_gts, coco_gt, batch_size, warm_up=False):
    
    model_predictor = load_model_predictor_from_name(model_id, device=device, return_batch=True)
    
    ds_id2label = {cat["id"]: cat["name"] for cat in loaded_json_gts["categories"]} 
    ds_label2id = {cat["name"]: cat["id"] for cat in loaded_json_gts["categories"]} 
    if model_predictor is None:
        print(f"Error while initializing model {model_id}")
        return {}
    
    print(f"\nEvaluation of model {model_id} started:")
    
    evaluator = evaluate.load("rafaelpadilla/detection_metrics", json_gt=loaded_json_gts, iou_type="bbox")

    # Prepare dataloader
    num_workers = multiprocessing.cpu_count()
    val_dataloader = DataLoader(coco_gt, batch_size=batch_size, 
                                num_workers=num_workers, 
                                collate_fn=model_predictor.collate_fn)

    # Warm up
    if warm_up:
        warm_up_batches = warm_up
        print(f"Warming up for {warm_up_batches} batches")
        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc="Evaluating batches")
            for idx, batch in enumerate(pbar, 1):
                predictions = model_predictor.predict_boxes(batch)
                if idx == warm_up_batches:
                    break

    start_time = time.time()
    
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
            
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print(f"Results model: {model_id}")
    results = evaluator.compute()
    print(f"Elapsed time: {elapsed_time} s")
    return results, elapsed_time



def benchmark_model_with_gpu_manager(model_id, hub_license, device, loaded_json_gts, coco_gt, batch_size, output_folder, warm_up=False):
    
    model_predictor = load_model_predictor_from_name(model_id, device=device, return_batch=True)
    
    ds_id2label = {cat["id"]: cat["name"] for cat in loaded_json_gts["categories"]} 
    ds_label2id = {cat["name"]: cat["id"] for cat in loaded_json_gts["categories"]} 
    if model_predictor is None:
        print(f"Error while initializing model {model_id}")
        return {}
    
    print(f"\nEvaluation of model {model_id} started @ {device}:")
    
    evaluator = evaluate.load("rafaelpadilla/detection_metrics", json_gt=loaded_json_gts, iou_type="bbox")

    # Prepare dataloader
    val_dataloader = DataLoader(coco_gt, batch_size=batch_size, 
                                collate_fn=model_predictor.collate_fn)
    # Warm up
    if warm_up:
        warm_up_batches = warm_up
        print(f"Warming up for {warm_up_batches} batches")
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader, 1):
                predictions = model_predictor.predict_boxes(batch)
                if idx == warm_up_batches:
                    break

    start_time = time.time()
    
    # Loop over the dataset
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
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
            
            if (idx % 100) == 0:
                perc_concluded = idx/len(val_dataloader)
                print(f"Model {model_id} concluded {100*perc_concluded:.2f}% @ GPU: {device}")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print(f"Results model: {model_id}")
    results = evaluator.compute()
    print(f"Elapsed time: {elapsed_time} s")
    
    results["iou_bbox"]["elapsed_time"] = elapsed_time
    results["iou_bbox"]["mean_time_per_sample"] = elapsed_time/len(coco_gt)
    results["iou_bbox"]["estimated_fps"] = len(coco_gt)/elapsed_time
    results["iou_bbox"]["hub_license"] = hub_license
    
    # Save
    str_model_id = model_id.replace("/","@")
    date_time = get_current_date_time()
    fp = output_folder / f"{str_model_id}@@{date_time}.json"
    with open(fp, "w") as file:
        json.dump(results, file, indent=4)
    print(f"File saved with success: {fp}")

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

def remove_evaluated_models(models_to_benchmark, output_folder):
    ret = []
    for model in models_to_benchmark:
        # Get info to save
        model_id = model["model_id"]
        str_model_id = model_id.replace("/","@")
        
        # If file with this name exists, skip
        matching_files = list(output_folder.glob(str_model_id+"*.json"))
        if len(matching_files) > 0:
            continue
        ret.append(model)
    return ret

def main(path_models_csv: Path, output_folder: Path, dataset_name: str = "rafaelpadilla/coco2017", batch_size = 1, use_gpu_manager: bool = False):
    
    path_models_csv = Path(path_models_csv)
    output_folder = Path(output_folder)
    
    # Get benchmarking dataset
    benchmark_dataset = load_dataset(dataset_name, split="val")
    target_classes = benchmark_dataset.features["objects"].feature["label"].names
    
    # Read models from csv
    df = pd.read_csv(path_models_csv) 
    
    models_to_benchmark = []
    # Get the models which have the target object classes
    for _, row in df.iterrows():
        
        output_classes_dict = ast.literal_eval(row["output_classes"])
        
        if not are_classes_comparable(target_classes, output_classes_dict):
            print(f"Skipping model {row['id']} (Classes are not comparable)")
        elif row["author"] in AUTHORS_TO_SKIP:
            print(f"Skipping model {row['id']} (Author)")
        elif row["id"] in MODELS_TO_SKIP:
            print(f"Skipping model {row['id']} (Id)")
        else:
            models_to_benchmark.append({"model_id": row["id"], "hub_license": row["hub_license"]})
    
    # Create a JSON in memory that represents the COCO dataset
    ids_mapping, loaded_json_gts = create_json_COCO_format(benchmark_dataset, round_approx=2)
    
    # Create ground-truth dataset based on the JSON
    coco_gt = COCODataset(loaded_json_gts, ids_mapping, benchmark_dataset)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Remove models that have been already evaluated
    print(f"Before: {len(models_to_benchmark)}")
    models_to_benchmark = remove_evaluated_models(models_to_benchmark, output_folder)
    print(f"After: {len(models_to_benchmark)}")
    
    if use_gpu_manager:
        # Create MultiGPUManager object
        manager = MultiGPUManager()

        kwargs = {
        "loaded_json_gts": loaded_json_gts, 
        "coco_gt": coco_gt, 
        "batch_size": batch_size,
        "warm_up": False, 
        "output_folder": output_folder
        }
        _ = manager.run(target_fn=benchmark_model_with_gpu_manager, iterable_items=models_to_benchmark, kwargs = kwargs)
    else:
        # Define default GPU: one model at a time
        device = "cuda:0"

        for model in models_to_benchmark:
            model_id = model["model_id"]
            try:
                metrics, elapsed_time = benchmark_model(model_id = model_id,
                                        device=device, 
                                        loaded_json_gts=loaded_json_gts, 
                                        coco_gt=coco_gt,
                                        batch_size=batch_size, 
                                        warm_up = 20)
                metrics["iou_bbox"]["elapsed_time"] = elapsed_time
                metrics["iou_bbox"]["mean_time_per_sample"] = elapsed_time/len(coco_gt)
                metrics["iou_bbox"]["estimated_fps"] = len(coco_gt)/elapsed_time
                metrics["iou_bbox"]["hub_license"] = model["hub_license"]
                # Save
                str_model_id = model_id.replace("/","@")
                date_time = get_current_date_time()
                fp = output_folder / f"{str_model_id}@@{date_time}.json"
                with open(fp, "w") as file:
                    json.dump(metrics, file, indent=4)
            except Exception as e:
                print(f"{e}: Error model {model_id}")

if __name__ == '__main__':
    Fire(main)  
   
