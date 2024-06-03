from pathlib import Path

import numpy as np
import pandas as pd

from utils.transcript import convert_turnwise

DATA_DIR = Path("/home/matsuura/Development/app/feature_extraction_api/experiment/data")

MONOLOGUE_TASK = ["Arg_Oly", "Cartoon", "RtSwithoutRAA", "RtSwithRAA"]
DIALOGUE_TASK = ["WoZ_Interview"]

PUNCTUATIONS = [".", ",", "!", "?"]

def str_2_df_transcript(str_transcript: str) -> pd.DataFrame:
    for punct in PUNCTUATIONS:
        str_transcript = str_transcript.replace(punct, " ")

    while "  " in str_transcript:
        str_transcript = str_transcript.replace("  ", " ")

    if str_transcript[0] == " ":
        str_transcript = str_transcript[1:]
    if str_transcript[-1] == " ":
        str_transcript = str_transcript[:-1]

    words = str_transcript.lower().split(" ")
    pad_list = np.ones_like(words)
    
    data = np.array([pad_list, pad_list, pad_list, words]).T
    df_transcript = pd.DataFrame(
        data,
        columns=["start_time", "end_time", "type", "text"]
    )

    return df_transcript

# 1. convert Google ASR lab file to csv format
for task in MONOLOGUE_TASK:
    load_dir = DATA_DIR / f"{task}/13_ASR_Google"
    for lab_path in load_dir.glob("*.lab"):
        save_path = load_dir / f"{lab_path.stem}_auto.csv"
        if save_path.exists():
            continue

        with open(lab_path, "r") as f:
            str_transcript = f.readline()
        df_transcript = str_2_df_transcript(str_transcript)     

        df_transcript.to_csv(save_path, index=False)

# 2. convert Google ASR csv file to csv format
for task in DIALOGUE_TASK:
    load_dir = DATA_DIR / f"{task}/13_ASR_Google"
    for csv_path in load_dir.glob("*.csv"):
        if csv_path.stem.endswith("_auto"):
            continue

        df_manual = pd.read_csv(DATA_DIR / f"{task}/01_Manual_TextGrid/{load_dir.stem}.csv")
        df_manual = convert_turnwise(df_manual)

# 3. convert Whisper ASR csv to csv format
for task in MONOLOGUE_TASK + DIALOGUE_TASK:
    load_dir = DATA_DIR / f"{task}/14_ASR_Whisper"
    for csv_path in load_dir.glob("*.csv"):
        if csv_path.stem.endswith("_auto"):
            continue

        save_path = load_dir / f"{csv_path.stem}_auto.csv"
        if save_path.exists():
            continue

        
        df_whisper = pd.read_csv(csv_path)
        df_whisper = df_whisper[~df_whisper["text"].isna()]
        if df_whisper.empty:
            continue
        
        try:
            str_transcript = " ".join(df_whisper["text"])
            df_transcript = str_2_df_transcript(str_transcript)     

            df_transcript.to_csv(save_path, index=False)
        except:
            print(save_path)
            print(df_whisper)
            break