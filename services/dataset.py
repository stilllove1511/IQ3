import json
from datasets import load_dataset

def formatting_func(example):
    text = f"### Question\n{example['question']}\n\n### Answer\n{example['output']}"
    return text

class DataSet(object):
    def __init__(self, upload_id):
        self.upload_id = upload_id
        
        self.train_dataset = load_dataset('json', data_files=f'data/custom_dataset_{self.upload_id}.jsonl', split='train')
        self.validation_dataset = load_dataset('json', data_files=f'data/custom_dataset_{self.upload_id}.jsonl', split='validation')