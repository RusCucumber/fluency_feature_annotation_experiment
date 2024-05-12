import json
from pathlib import Path
from tqdm import tqdm
from utils.rev_asr import RevAsr, save_job
from resources.rev_asr_access_token import ACCESS_TOKEN

DATA_DIR = Path("/home/matsuura/Development/app/feature_extraction_api/experiment/data")
TASK = ["Arg_Oly", "Cartoon", "RtSwithoutRAA", "RtSwithRAA"]

def main():
    asr = RevAsr(ACCESS_TOKEN)

    for task in TASK:
        load_dir = DATA_DIR / f"{task}/03_FA_Audio_Transcript_Auto"
        save_dir = DATA_DIR / f"{task}/02_Rev_Transcript"

        n_speech = len(list(load_dir.glob("*.wav")))
        for audio_path in tqdm(load_dir.glob("*.wav"), desc=f"POST audio data from {task}", total=n_speech):
            save_path = save_dir / f"{audio_path.stem}.json"
            if save_path.exists():
                continue

            job = asr.submit(audio_path)
            save_job(job, save_path)
            
if __name__ == "__main__":
    main()