import os
import torch
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
        if "batch_size" in kwargs.keys():
            batch_size = kwargs["batch_size"]
        else:
            batch_size = 1000
        self.dataset = self.dataset.map(
            lambda example: operation(example, columns, **kwargs), 
            batched=kwargs["batched"],
            batch_size=batch_size
        )

    def shuffle(self, seed):
        self.dataset = self.dataset.shuffle(seed=seed)

    def filter_dataset(self, split: Union[str, None], column: str, value: Union[str, int]):
        if split:
            return self.dataset[split].filter(lambda example: example[column] == value)
        return self.dataset.filter(lambda example: example[column] == value)

    def get_split(self, split: str):
        return self.dataset[split]

    def get_split_names(self):
        return list(self.dataset.keys())

    def get_split_sizes(self):
        return {split: len(self.dataset[split]) for split in self.dataset.keys()}

    def subsample_split_classes(self, data, n):
        classes = [0, 1, 2]
        subsamples = []
        for class_ in classes:
            class_data = data.filter(lambda x: x["label"] == class_)
            subsamples.append(class_data.select(list(range(n))))
        
        return concatenate_datasets(subsamples)

    def subsample_split_groups(self, data, n, splits, seed):
        classes = [0, 1, 2]

        split_data = concatenate_datasets([data[split] for split in splits])
        subsamples = []
        for class_ in classes:
            class_data = data.filter(lambda x: x["label"] == class_).shuffle(seed=seed)
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


    def add_cartography_data(self, split_groups, path_prefixes, column_prefixes):
        split_sizes = self.get_split_sizes()
        for path_prefix, column_prefix in zip(path_prefixes, column_prefixes):
            for eval_split in split_groups.keys():
                path = path_prefix + f"_{eval_split}_cartography.pt"
                data = torch.load(path)
                offset = 0
                for split in split_groups[eval_split]:            
                    self.dataset[split] = self.dataset[split].map(
                        lambda example, idx: {
                            f"{column_prefix}_confidence": data[idx + offset, 0].item(),
                            f"{column_prefix}_variability": data[idx + offset, 1].item(),
                            f"{column_prefix}_correctness": data[idx + offset, 2].item()
                        }, with_indices=True
                    )

                    offset += split_sizes[split]


    def add_pvi_data(self, split_groups, path_prefixes, column_prefixes):
        split_sizes = self.get_split_sizes()
        for path_prefix, column_prefix in zip(path_prefixes, column_prefixes):
            for eval_split in split_groups.keys():
                path = path_prefix + f"_pvi_{eval_split}.pt"
                data = torch.load(path)
                offset = 0
                for split in split_groups[eval_split]:
                    self.dataset[split] = self.dataset[split].map(
                        lambda example, idx: {
                            f"{column_prefix}_pvi": data[idx + offset, 0].item(),
                            f"{column_prefix}_final_correct": data[idx + offset, 1].item()
                        }, with_indices=True
                    )
                    offset += split_sizes[split]
