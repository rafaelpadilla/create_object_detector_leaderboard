
![layout](https://github.com/rafaelpadilla/create_object_detector_leaderboard/blob/main/assets/leaderboard.png?raw=true "Title")

# Object Detection Leaderboard

This project provides a tool to run object detection models from 🤗 Hugging Face Hub with a specific dataset and compute their detection metrics. 

Models can only be compared if they are able to output the same classes. For example, a model that was trained to detect dogs breeds `{1: "bulldog", 2: "labrador", 3: "husky"}` is not compatible with a model trained to detect cats `{1: "siamese", 2: "persian", 3: "maine coon"}`.

Seen that, we need to group models whose output classes are compatible and compared.

## Install
1. Clone this repo: `git clone https://github.com/rafaelpadilla/object_detection_leaderboard`
2. Go to the project's folder: `cd object_detection_leaderboard`
3. Create an environment so you don't mess with libraries: `conda create -n leaderboard`
4. Activate environment: `conda activate leaderboard`
5. Install package: `pip install -e .`

### Step 1: Getting comparable models
The script `get_comparable_models.py` lists object detection models from the hub and outputs a csv file `object_detection_models-MM-DD-YYYY_HH_MM_SS.csv`. The column `output_classes` lists the classes that each model is able to detect.

To generate it, run the following command:
```
python scripts/get_comparable_models.py
```

### Step 2: Obtaining metrics with comparable models

To run the metrics of the models listed in `object_detection_models-MM-DD-YYYY_HH_MM_SS.csv`, we call `testing_dataset.py`:

To generate it, run the following command:
```
python scripts/evaluate_models.py --path_models_csv path_to/object_detection_models-MM-DD-YYYY_HH_MM_SS.csv --output_folder path_to/results_folder --dataset_name rafaelpadilla/coco2017
```

### Step 3: Compile all results in a single csv

Once the results of the desired models are computed, we run `compile_results.py` to create a unique csv files gathering the results of models.

```
python scripts/compile_results.py --dir_results output_folder path_to/results_folder --output_csv output_folder path_to/results_folder/results.csv
```
