from pathlib import Path
import pandas as pd

def transcript_generator(transcript_dir, filenames=[], suffix="csv", **kwargs):
    if not isinstance(transcript_dir, (Path, str)):
        raise TypeError(f"transcript_dir must be path like object.\n{type(transcript_dir)} is not supported.")

    if suffix == "csv":
        reader = pd.read_csv
    elif suffix == "tsv":
        reader = pd.read_table
    else:
        raise ValueError(f"suffix must be csv or tsv. {suffix} is not supported.")

    transcript_dir = Path(transcript_dir)

    if len(filenames) == 0:
        for filename in list(transcript_dir.glob(f"*.{suffix}")):
            yield reader(filename, **kwargs)
    else:
        for name in filenames:
            yield reader(transcript_dir / f"{name}.{suffix}", **kwargs)


def convert_turnwise(df, speaker_col="speaker", **kwargs):
    data = []
    columns = df.columns.values

    prev_speaker = df.at[0, speaker_col]
    utterances = []
    for _, row in df.iterrows():
        if prev_speaker == row.speaker:
            utterances.append(row)
        else:
            new_row = generate_new_row(utterances, columns, **kwargs)
            data.append(new_row)
            utterances = [row]
        prev_speaker = row.speaker

    new_row = generate_new_row(utterances, columns, **kwargs)
    data.append(new_row)

    return pd.DataFrame(data, columns=columns)


def generate_new_row(utterances, columns, end_time_col="end_time", transcript_col="transcript"):
    row = []
    for col in columns:
        if col == end_time_col:
            val = getattr(utterances[-1], col)
        elif col == transcript_col:
            val = " ".join([getattr(u, col) for u in utterances if str(getattr(u, col)) != "nan"])
        else:
            val = getattr(utterances[0], col)
            
        row.append(val)

    return row