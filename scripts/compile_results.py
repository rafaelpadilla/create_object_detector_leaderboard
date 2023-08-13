import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from fire import Fire

def main(dir_results: Path, output_csv: Path):
    dir_results = Path(dir_results)
    output_csv = Path(output_csv)
    
    files = list(dir_results.glob("*.json"))

    assert dir_results.exists(), f"Directory with results {dir_results} not found."
    assert len(files) > 0, f"Directory with results {dir_results} does not contain any result."
    assert not output_csv.exists(), f"Output file {output_csv} already exists."

    # Find duplicated model names
    model_names = []
    for f in files:
        fn = f.stem
        model_name = fn[0:fn.find("@@")].replace("@","/")
        model_names.append(model_name)
    repeated = set([x for x in model_names if model_names.count(x) > 1])
    total_repeated = len(repeated)
    str_repeated = ", ".join(repeated)
    assert total_repeated == 0, f"There are {total_repeated} duplicated models: {str_repeated}"

    data_frames = []
    pbar = tqdm(files, desc="Compiling results")
    for f in pbar:
        fn = f.stem
        model_name = fn[:fn.find("@@")].replace("@","/")
        dict_results = json.load(open(f, "r"))["iou_bbox"]
        df = pd.DataFrame.from_dict(dict_results, orient="index").T
        df.insert(loc=0, column='model', value=model_name)
        data_frames.append(df)

    df = pd.concat(data_frames, axis=0)
    df.sort_values("AP-IoU=0.50:0.95-area=all-maxDets=100", ascending=False, inplace=True)
    df.to_csv(output_csv, index=False)
    
    print(f"File {output_csv} generated with success!")


if __name__ == "__main__":
    Fire(main)
    