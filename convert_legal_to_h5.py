import h5py
import os
import json

from pathlib import Path
from glob import glob

BASE_PATH = Path("./data/legal-dataset/ALL/test-data")

filenames = []

for file in glob(os.path.join(BASE_PATH, "judgement", "*.txt")):
    filenames.append(file)


file_pairs = []
for file in filenames:
    judgement_file = file
    summary_file = judgement_file.replace("judgement", "summary")
    file_pairs.append((judgement_file, summary_file))


data = []

for fn1, fn2 in file_pairs:
    with open(fn1, "r") as f1, open(fn2, "r") as f2:
        judgement = f1.read()
        summary = f2.read()
        t = {"article": judgement, "abstract": summary}

        data.append(t)

formatted_data = [json.dumps(i).encode() for i in data]

hf = h5py.File("./data/legal-dataset/legal.test.h5df", "w")

hf.create_dataset("dataset", data=formatted_data)
