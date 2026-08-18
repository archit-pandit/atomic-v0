import os
import pandas
import pyarrow.parquet as pq
from tqdm import tqdm

def parquet_files_in_dir(directory):
    """
    Returns a list of all parquet files in the given directory and its subdirectories.
    """
    parquet_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    return parquet_files

FOLDER_PATH = 'E:/datasets'
OUTPUT_FILE_TRAIN = 'output_train.txt'
OUTPUT_FILE_VALID = 'output_valid.txt'
VOCAB_FILE = 'vocab.txt'

files = parquet_files_in_dir(FOLDER_PATH)

TOTAL_FILES = len(files)
SPLIT_INDEX = int(TOTAL_FILES * 0.8)

files_train = files[:SPLIT_INDEX]
files_valid = files[SPLIT_INDEX:]

vocab = set()

with open(OUTPUT_FILE_TRAIN, 'w', encoding='utf-8') as f:
    for count, filename in enumerate(tqdm(files_train,
                                          total=len(files_train),
                                          desc='Processing training files')):
        file_path = os.path.join(FOLDER_PATH, filename)
        table =  pq.read_table(file_path)

        for row in table.to_pandas().itertuples(index=False):
            text = row[0]

            if isinstance(text, str):
                f.write(text)
                characters = set(text)
                vocab.update(characters)

with open(OUTPUT_FILE_VALID, 'w', encoding='utf-8') as f:
    for count, filename in enumerate(tqdm(files_valid,
                                          total=len(files_valid),
                                          desc='Processing validation files')):
        file_path = os.path.join(FOLDER_PATH, filename)
        table =  pq.read_table(file_path)

        for row in table.to_pandas().itertuples(index=False):
            text = row[0]

            if isinstance(text, str):
                f.write(text)
                characters = set(text)
                vocab.update(characters)

with open(VOCAB_FILE, 'w', encoding='utf-8') as vocab_f:
    for char in vocab:
        vocab_f.write(char + '\n')
