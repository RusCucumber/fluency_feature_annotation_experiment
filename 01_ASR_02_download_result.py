import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from utils.rev_asr import RevAsr
from resources.rev_asr_access_token import ACCESS_TOKEN

DATA_DIR = Path("/home/matsuura/Development/app/feature_extraction_api/experiment/data")
TASK = ["Arg_Oly", "Cartoon", "RtSwithoutRAA", "RtSwithRAA"]

def main():
    asr = RevAsr(ACCESS_TOKEN)

    for task in TASK:
        data_dir = DATA_DIR / f"{task}/02_Rev_Transcript"

        n_job = len(list(data_dir.glob("*.json")))
        for job_path in tqdm(data_dir.glob("*.json"), desc=f"Request transcript in {task}", total=n_job):
            save_path = data_dir / f"{job_path.stem}_long.csv"
            if save_path.exists():
                continue

            with open(job_path, "r") as f:
                response = json.load(f)
                job_id = response["id"]

            try:
                if asr.is_job_completed(job_id):
                    df_rev = asr.load_transcription(job_id)
                    df_rev.to_csv(save_path)
            except RuntimeError:
                print(f"job {job_id} failed: {response['failure']}")


if __name__ == "__main__":
    main()