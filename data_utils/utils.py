import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union
from load_data import get_dataset
from load_utils.load_configs import LOAD_CONFIGS
from spacy.lang.en import English
from transformers import AutoTokenizer
from scipy.stats import kde
from datasets.fingerprint import Hasher

def get_sequence_length(sample, columns):
    return {
        column + "_length": len(sample[column].split(" "))
        for column in columns
    }

def get_sequence_length_batch(examples, columns, **kwargs):
    return {
        column + kwargs["affix"]: kwargs["tokenizer_fn"](examples[column], return_length=True)["length"]
        for column in columns
    }

def get_sequence_length_example(sample, columns, **kwargs):
    return {
        column + kwargs["affix"]: len(kwargs["tokenizer_fn"](sample[column]))
        for column in columns
    }

def tokenize_spacy_no_punct(text: str, tokenizer_fn) -> List[str]:
    return [token.text.lower() for token in tokenizer_fn(text) if not token.is_punct]

def remove_punct(sequences: Union[List[str], str]) -> Union[List[str], str]:
    if type(sequences) == str:
        return re.sub(r'^\w\s', '', sequences)
    return [re.sub(r'[^\w\s]', '', sequence) for sequence in sequences]

def tokenize_whitespace(example):
    return example.lower().split(" ")

def get_n_grams_batch(examples, columns, **kwargs):
    tokenizer_fn = kwargs["tokenizer_fn"]
    n = kwargs["n"]
    results = {}
    for column in columns:
        sequences = tokenizer_fn(remove_punct(examples[column]))["input_ids"]
        column_name = column + kwargs["affix"]
        results[column_name] = [
            [sequence[i:i+n] for i in range(len(sequence) - n + 1)] 
            for sequence in sequences
        ]
    return results

def get_n_grams(example, columns, **kwargs):
    tokenizer_fn = kwargs["tokenizer_fn"]
    n = kwargs["n"]
    results = {}
    for column in columns:
        sequence = tokenizer_fn(remove_punct(example[column]))
        column_name = column + kwargs["affix"]
        results[column_name] = [" ".join(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    return results

class DataAnalyzer:

    def __init__(self, processor: DataProcessor, save_dir:str):
        self.processor = processor
        self.save_dir = save_dir
        self.class_names = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    def plot_sequence_lengths(self, column):
        for split in list(self.processor.dataset.keys()):
            plt.figure(figsize=(12, 10))
            for class_ in range(3):
                samples = self.processor.filter_dataset(split, "label", class_)
                density = kde.gaussian_kde(samples[column])
                #x = list(range(max(samples[column])))
                x = list(range(40))
                y = density(x)
                plt.plot(x, y, label=self.class_names[class_])

            plt.xticks(x, fontsize=8)
            plt.xlabel("length")
            plt.ylabel("density")
            plt.legend()
            
            plt.savefig(self.save_dir + f"/_{split}_{column}.png")
            plt.clf()

def get_n_unique_n_grams(samples, columns):
    unique_n_grams = set()
    for column in columns:
        unique_n_grams.update(set(tuple(n_gram) for sample in samples[column] for n_gram in sample))
    return len(unique_n_grams)

def get_unique_top_n_grams(dataset_name, tokenizer, tokenizer_name, n, batched):
    processor = DataProcessor(LOAD_CONFIGS[dataset_name])
    columns = ["premise", "hypothesis"]
    if "gbda" in dataset_name:
        columns = ["adversarial_premise", "adversarial_hypothesis"]
   
    if batched:
        operation = get_n_grams_batch
    else:
        operation = get_n_grams

    results = {i:{} for i in range(1, n + 1)}
    for i in range(1, n + 1):
        affix = f"_{i}_grams_{tokenizer_name}"
        processor.apply_operation(columns, operation, batched, tokenizer_fn=tokenizer, affix=affix, n=i)
        for split in list(processor.dataset.keys()):
            samples = processor.dataset[split]
            results[i][split] = get_n_unique_n_grams(samples, [col + affix for col in columns])

    return results

def get_all_top_unique_n_grams(tokenizer, tokenizer_name, n, batched):
    datasets = ["snli", "mnli", "anli", "gbda_premise", "gbda_hypothesis", "cad_snli_combined",
                "cad_snli_original", "cad_snli_revised_combined", "cad_snli_revised_hypothesis",
                "cad_snli_revised_premise"]

    for dataset in datasets:
        results = get_unique_top_n_grams(dataset, tokenizer, tokenizer_name, 2, batched)
        export_top_unique_n_grams(results, "results/" + dataset + f"/unique_n_grams_{tokenizer_name}.csv")


def export_top_unique_n_grams(results, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["gram"] + list(results[1].keys()))
        for n, splits in results.items():
            writer.writerow([n] + [count for count in splits.values()])

def plot_nli_sequence_lengths(dataset_name, tokenizer, tokenizer_name):
    processor = DataProcessor(LOAD_CONFIGS[dataset_name])
    
    columns = ["premise", "hypothesis"]
    if "gbda" in dataset_name:
        columns.extend(["adversarial_premise", "adversarial_hypothesis"])
    
    processor.apply_operation(columns, get_sequence_length_batch, True, tokenizer_fn=tokenizer, affix=f"_sequence_length_{tokenizer_name}", return_length=True)
    
    analyzer = DataAnalyzer(processor, "results/" + dataset_name)
    for column in columns:
        analyzer.plot_sequence_lengths(f"{column}_sequence_length_{tokenizer_name}")
        analyzer.plot_sequence_lengths(f"{column}_sequence_length_{tokenizer_name}")

def plot_all_nli_sequence_lengths(tokenizer, tokenizer_name):
    datasets = ["snli", "mnli", "anli", "gbda_premise", "gbda_hypothesis", "cad_snli_combined",
                "cad_snli_original", "cad_snli_revised_combined", "cad_snli_revised_hypothesis",
                "cad_snli_revised_premise"]

    for dataset in datasets:
        plot_nli_sequence_lengths(dataset, tokenizer, tokenizer_name)

def get_n_examples():
    datasets = ["snli", "mnli", "anli", "gbda_premise", "gbda_hypothesis", "cad_snli_combined",
                "cad_snli_original", "cad_snli_revised_combined", "cad_snli_revised_hypothesis",
                "cad_snli_revised_premise"]

    for dataset in datasets:
        processor = DataProcessor(LOAD_CONFIGS[dataset])
        for split in processor.dataset.keys():
            n = len(processor.dataset[split])
            print(f"{dataset}\t{split}: {n} examples")

if __name__ == "__main__":
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    #get_all_top_unique_n_grams(tokenize_whitespace, "whitespace", 2, False)
    get_n_examples()

    #results = get_unique_top_n_grams("snli", tokenize_whitespace, "whitespace", 2, False)

    #results = get_unique_top_n_grams("snli", tokenizer, "bert", 2)
    #export_top_unique_n_grams(results, "results/snli_unique_n_grams_bert.csv")

    #processor = DataProcessor(LOAD_CONFIGS["snli"])
    #processor.apply_operation(["hypothesis", "premise"], get_n_grams_batch, True, tokenizer_fn=tokenizer, affix="_1_grams_bert", n=1)
    #print(processor.dataset["train"][0])
    #plot_all_nli_sequence_lengths(tokenizer, "bert")
