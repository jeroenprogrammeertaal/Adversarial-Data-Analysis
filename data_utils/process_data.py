import os
from typing import List, Union
from .load_data import get_dataset
from datasets import concatenate_datasets

class DataProcessor:

    def __init__(self, load_config: dict, seed: int = 42, shuffle: bool = False):
        self.dataset = get_dataset(load_config["dataset"], **load_config)
        self.seed = seed

        if shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed)

    def apply_operation(self, columns: List[str], operation, **kwargs):
        self.dataset = self.dataset.map(
            lambda example: operation(example, columns, **kwargs), 
            batched=kwargs["batched"]
        )

    def filter_dataset(self, split: Union[str, None], column: str, value: Union[str, int]):
        if split:
            return self.dataset[split].filter(lambda example: example[column] == value)
        return self.dataset.filter(lambda example: example[column] == value)

    def get_split(self, split: str):
        return self.dataset[split]

    def get_split_names(self):
        return list(self.dataset.keys())

    def subsample_split_classes(self, data, n):
        classes = [0, 1, 2]
        subsamples = []
        for class_ in classes:
            class_data = data.filter(lambda x: x["label"] == class_)
            subsamples.append(class_data.select(list(range(n))))
        
        return concatenate_datasets(subsamples)
    
    def filter_column_function(self, split, func):
        return self.dataset[split].filter(func)
    
    def get_split_data(self, split, correctness="both"):
        if correctness == "both":
            return self.dataset[split]
        
        if correctness == "incorrect":
            return self.filter_column_function(split, lambda x: len(x["reason"]) > 0)
            #return self.dataset[split].filter(lambda x: len(x["reason"]) > 0)
        
        if correctness == "correct":
            return self.filter_column_function(split, lambda x: len(x["reason"]) == 0)
            #return self.dataset[split].filter(lambda x: len(x["reason"]) == 0)

        if correctness == "generated":
            return self.filter_column_function(split, lambda x: x["genre"] == "generated")

        if correctness == "generated_revised":
            return self.filter_column_function(split, lambda x: x["genre"] == "generated_revised")
