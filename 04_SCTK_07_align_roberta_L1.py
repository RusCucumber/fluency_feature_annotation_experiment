from typing import Optional, Union, List, Tuple, Generator
from pathlib import Path
import subprocess
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("/home/matsuura/Development/app/feature_extraction_api/experiment/data")

TASK = ["WoZ_Interview"]
IGNORE_TAGS = [
    ["<CI>", "<CE>", "<FILLER>"], # for disfluency alignment
    ["<CI>", "<CE>"], # for disfluency alignment 2
    ["<DISFLUENCY>", "<FILLER>"] # for pause location alignment
]
FILLER = {"uh", "ah", "um", "mm", "hmm", "oh", "mm-hmm", "er", "mhm", "uh-huh", "er", "erm", "huh", "uhu", "mmhmm", "uhhuh"}
SCTK_COMMAND = ["docker", "run", "-it", "-v", f"{str(DATA_DIR)}:/var/sctk", "sctk", "sclite", "-i", "wsj", "-r", "ref.txt", "-h", "hyp.txt", "-p"]

def sctk_input_csv_path_generator(task: str) -> Generator[Tuple[Path, Path], None, None]:
    load_dir = DATA_DIR / f"{task}/10_SCTK_Inputs"

    for manu_csv_path in load_dir.glob("*_manu.csv"):
        filename = manu_csv_path.stem.removesuffix("_manu")
        auto_csv_path = load_dir / f"{filename}_auto_roberta_L1.csv"

        yield manu_csv_path, auto_csv_path

def write_ref_hyp_txt_files(
        df_ref: pd.DataFrame, 
        df_hyp: pd.DataFrame, 
        ignore_tag: Optional[Union[str, List[str]]]=None
) -> None:
    
    for filler in FILLER:
        mask_filler_ref = (df_ref["text"] == filler)
        df_ref = df_ref[~mask_filler_ref]

        mask_filler_hyp = (df_hyp["text"] == filler)
        df_hyp = df_hyp[~mask_filler_hyp]
    
    ref = df_ref["text"].values.astype(str)
    hyp = df_hyp["text"].values.astype(str)

    ref = " ".join(ref) + "\n"
    hyp = " ".join(hyp) + "\n"

    if ignore_tag is not None:
        if isinstance(ignore_tag, str):
            ignore_tag = [ignore_tag]
        
        for tag in ignore_tag:
            ref = ref.replace(tag, "")
            hyp = hyp.replace(tag, "")

    with open(DATA_DIR / f"ref.txt", "w") as f:
        f.write(ref)
    with open(DATA_DIR / f"hyp.txt", "w") as f:
        f.write(hyp)
    

if __name__ == "__main__":
    for task in TASK:
        for ref_csv_path, hyp_csv_path in tqdm(sctk_input_csv_path_generator(task), desc=f"[{task}] Aligning..."):
            df_ref = pd.read_csv(ref_csv_path, na_values=["", " "], keep_default_na=False)
            df_ref = df_ref.sort_values(["start_time", "type"])
            
            df_hyp = pd.DataFrame([], columns=["text"])
            if hyp_csv_path.exists():
                df_hyp = pd.read_csv(hyp_csv_path, na_values=["", " "], keep_default_na=False)
                df_hyp = df_hyp.sort_values(["start_time", "type"])

            for ignore_tag in IGNORE_TAGS:
                filename = f"{hyp_csv_path.stem.removesuffix('_auto')}_ignore_{'-'.join(ignore_tag)}"
                save_path = DATA_DIR / f"{task}/11_SCTK_Outputs/{filename}_roberta_L1.txt"
                if save_path.exists():
                    continue

                write_ref_hyp_txt_files(df_ref, df_hyp, ignore_tag)

                with open(save_path, "w") as f:
                    cp = subprocess.run(SCTK_COMMAND, stdout=f)