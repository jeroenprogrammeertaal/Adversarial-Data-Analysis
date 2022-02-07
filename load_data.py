import os
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union
from datasets import load_dataset, load_from_disk
from scipy.stats import kde


class DataPipeline:

    def __init__(self, 
            name: Union[str, None] = None,
            load_script: Union[str, None] = None,
            path: Union[str, None] = None,
            num_classes: Union[int, None] = None
        ):

        if path:
            self.data_dict = load_from_disk(path)
        elif load_script:
            self.data_dict = load_dataset(load_script, name)
        else:
            assert name is not None
            self.data_dict = load_dataset(name)

        self.splits = list(self.data_dict.keys())
        self.num_classes = num_classes
        self.class_names = {
                    0 : "entailment",
                    1 : "neutral",
                    2 : "contradiction"
                }

    def __getitem__(self, key):
        return self.data_dict[key]

    def add_sequence_lengths(self, columns: List[str]) -> None:
        self.data_dict = self.data_dict.map(
            lambda x: {
                c + "_length": len(x[c].split(" "))
                for c in columns
            }
        )

    def get_class_samples(self, class_: int, split: str):
        return self.data_dict[split].filter(lambda x: x["label"] == class_)

        
    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.data_dict.save_to_disk(path)
        

class DataAnalyse:

    def __init__(self, datapipeline):
        self.pipeline = datapipeline

    def plot_class_lengths(self, 
                        splits: List[str], 
                        column: str, 
                        save_dir: str) -> None:
        for split in splits:
            for c in range(self.pipeline.num_classes):
                samples = self.pipeline.get_class_samples(c, split)
                density = kde.gaussian_kde(samples[column])
                x = list(range(100))
                y = density(x)
                plt.plot(x, y, label=self.pipeline.class_names[c])

            plt.xticks(x)
            plt.xlabel("length")
            plt.ylabel("density")
            plt.legend()
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            
            plt.savefig(save_dir + "/" + column + "_" + split)

            plt.clf()

# TODO:
#   - load other datasets
#   - Diversity Metrics
#   - Complexity Metrics


if __name__ == "__main__":

    mrqa = DataPipeline(load_script="load_contrast_sets.py", name="boolq")
    mrqa.add_sequence_lengths(["paragraph", "question", "original_question"])
    mrqa.save("data/contrast_sets/boolq")
    #mrqa_analyse = DataAnalyse(mrqa)
    print(mrqa["test"][0])

    #cad_snli_analyse.plot_class_lengths(cad_snli.splits, 
    #        "premise_length", 
    #        "results/anli")
