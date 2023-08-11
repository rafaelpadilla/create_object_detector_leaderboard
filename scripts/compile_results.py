import json
import pandas as pd
from pathlib import Path

dir_results = Path("/home/rafael/hf/leaderboard_results/")
files = list(dir_results.glob("*.json"))

data_frames = []
for f in files:
    fn = f.stem
    print(f"Reading file {fn}")
    model_name = fn[:fn.find("@@")].replace("@","/")
    dict_results = json.load(open(f, "r"))["iou_bbox"]
    df = pd.DataFrame.from_dict(dict_results, orient="index").T
    df.insert(loc=0, column='model', value=model_name)
    data_frames.append(df)

df = pd.concat(data_frames, axis=0)
df.to_csv("all_results.csv", index=False)

