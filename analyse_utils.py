import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pyplot_themes as themes
import csv
import numpy as np
import torch
import random
import pickle
import nltk

from typing import List, Union
from collections import defaultdict, deque
from data_utils.process_data import DataProcessor
from data_utils.load_configs import LOAD_CONFIGS
from scipy.stats import kde
from adjustText import adjust_text
from sentence_transformers import SentenceTransformer, util
from datasets import concatenate_datasets
from tqdm import tqdm


def map_labels(examples, column, **kwargs):
    labels = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }
    return {
        "label": [labels[example] for example in examples[column]]
    }

def remove_punct(sequences: Union[List[str], str]) -> Union[List[str], str]:
    if type(sequences) == str:
        return re.sub(r'[^\w\s]', '', sequences)
    return [re.sub(r'[^\w\s]', '', sequence) for sequence in sequences]

def whitespace_tokenizer(examples: dict, columns: List[str], **kwargs):
    # NOTE: Reintroduce .lower()
    if not kwargs["batched"]:
        return {
            column + kwargs["affix"]: remove_punct(examples[column].lower()).split(" ")
            for column in columns
        }
    return {
        column + kwargs["affix"]: [remove_punct(example.lower()).split(" ") for example in examples[column]]
        for column in columns
    }

def get_example_lengths(examples: dict, columns: [List[str]], **kwargs):
    if not kwargs["batched"]:
        return {
            column + kwargs["affix"]: len(examples[column])
            for column in columns
        }
    return {
        column + kwargs["affix"]: [len(example) for example in examples[column]]
        for column in columns
    }

def get_example_n_grams(examples: dict, columns: List[str], **kwargs):
    
    def extract_grams(sequences: List[List[str]], n):
        return [
            [sequence[i:i+n] for i in range(len(sequence) - n + 1)]
            for sequence in sequences
        ]

    if not kwargs["batched"]:
        return {
            column + kwargs["affix"]: extract_grams([examples[column]], kwargs["n"])[0]
            for column in columns
        }
    
    return {
        column + kwargs["affix"]: extract_grams(examples[column], kwargs["n"])
        for column in columns
    }

def get_example_overlap(examples, columns, **kwargs):
    """ Calculates portion of tokens in column_2 which is also in column_1."""

    def extract_overlaps(sequences_1, sequences_2):
        return [
            len(set(sequences_1[i]) & set(sequences_2[i])) / len(set(sequences_2[i]))
            for i in range(len(sequences_1))
        ]

    if not kwargs["batched"]:
        return {
            kwargs["affix"] : extract_overlaps([examples[columns[0]]], [examples[columns[1]]])[0]
        }
    return {
        kwargs["affix"]: extract_overlaps(examples[columns[0]], examples[columns[1]])
    }

def get_example_cosine_sim(examples, columns, **kwargs):

    def extract_cos_sim(sequences_1, sequences_2, model):
        return torch.diagonal(
            util.cos_sim(
                model.encode(sequences_1),
                model.encode(sequences_2)
            ), 0
        ).tolist()
    
    results = extract_cos_sim(examples[columns[0]], examples[columns[1]], kwargs["model"])
    if not kwargs["batched"]:
        return {
            kwargs["affix"] : results[0]
        }
    return {
        kwargs["affix"] : results
    }

def get_example_yngve(examples, columns, **kwargs):
    
    def avg_yngve_score(doc):
        score = 0
        for sent in doc.sents:
            root_distance = {}
            stack = deque()

            root = list(sent._.constituents)[0]
            stack.append(root)

            while len(stack) > 0:
                constituent = stack.pop()
                if len(list(constituent._.children)) == 0:
                    root_distance[constituent] = len(stack)

                for child in list(constituent._.children)[::-1]:
                    if str(child) not in [".", ",", ";", "!", "?", "'", "`"]:
                        stack.append(child)

            score += np.mean(list(root_distance.values()))
        return score / len(list(doc.sents))

    def parse_texts(texts, model):
        if type(texts) == str:
            texts = [texts]

        return model.pipe(texts)

    if not kwargs["batched"]:
        return {
            col + "_yngve": avg_yngve_score(parse_texts(examples[col], kwargs["model"]))
            for col in columns
        }
    return {
        col + "_yngve": [avg_yngve_score(x) for x in parse_texts(examples[col], kwargs["model"])]
        for col in columns
    }

def get_parse_tree_height(examples, columns, **kwargs):

    def get_height(doc):
        
        def walk_tree(node, depth):
            if len(list(node._.children)) > 0:
                return max([walk_tree(child, depth + 1) for child in node._.children])
            return depth

        score = 0
        for sent in doc.sents:
            root = list(sent._.constituents)[0]
            score += walk_tree(root, 0)
        return score / len(list(doc.sents))

    def parse_texts(texts, model):
        if type(texts) == str:
            texts = [texts]
        return model.pipe(texts)
    
    if not kwargs["batched"]:
        return {
            col + "_parse_height": get_height(parse_texts(examples[col], kwargs["model"]))
            for col in columns
        }
    return {
        col + "_parse_height": [get_height(x) for x in parse_texts(examples[col], kwargs["model"])]
        for col in columns
    }


def get_example_hans_heuristics(examples, columns, **kwargs):
    # Columns[0]: Premise tokens
    # Columns[1]: Hypothesis tokens
    # Columns[2]: Premise texts
    # Columns[3]: Hypothesis texts

    def find_lexical_heuristic(examples, columns):
        # Columns[0]: Premise tokens
        # Columns[1]: Hypothesis tokens
        # returns 1 if all tokens in columns[1] occur in columns[0] else 0
        if type(examples) == str:
            examples = [examples]

        # Temporary weird tokenization. This reproduces results from original paper.
        #punct = [".", "?", "!"]
        #premise_tokens = []
        #hypothesis_tokens = []
        #for premise, hypothesis in zip(examples[columns[2]], examples[columns[3]]):
        #    premise_tokens.append([t for t in premise.lower().split() if t not in punct])
        #    hypothesis_tokens.append([t for t in hypothesis.lower().split() if t not in punct])
        
        #return [
        #    1 if set(col_1).issubset(col_0) else 0
        #    for col_1, col_0 in zip(hypothesis_tokens, premise_tokens)
        #]

        # Better version I think.
        return [
            1 if set(col_1).issubset(col_0) else 0
            for col_1, col_0 in zip(examples[columns[1]], examples[columns[0]])
        ]

    def find_subsequence_heuristic(examples, columns):
        # Columns[0]: Premise tokens
        # Columns[1]: Hypothesis tokens
        # Returns 1 if all tokens in columns[1] are a contigues subsequence in columns[0] else 0
        if type(examples) == str:
            examples = [examples]
        return [
            1 if " ".join(col_1) in " ".join(col_0) else 0
            for col_1, col_0 in zip(examples[columns[1]], examples[columns[0]])
        ]

    def find_constituence_heuristic(examples, idx, columns, model):
        # Columns[0]: premise texts
        # Columns[1]: Hypothesis tokens
        results = [0 for _ in range(len(examples[columns[0]]))]
        if type(examples) == str:
            examples = [examples]

        col_1_docs = model.pipe([examples[columns[0]][i] for i in idx])
        col_2_texts = [" ".join(tokens) for tokens in [examples[columns[1]][i] for i in idx]]
        for i, doc, text in zip(idx, col_1_docs, col_2_texts):
            results[i] = 1 if text in [c.text.lower() for c in list(doc.sents)[0]._.constituents] else 0
        return results

    results_lexical = find_lexical_heuristic(examples, columns)
    results_subsequence = find_subsequence_heuristic(examples, columns)
    subsequence_ids = [i for i in range(len(results_subsequence)) if results_subsequence[i] == 1]
    results_const = find_constituence_heuristic(examples, subsequence_ids, [columns[2], columns[1]], kwargs["model"])
    if not kwargs["batched"]:
        return {
            "lexical_heuristic": results_lexical[0],
            "subsequence_heuristic": results_subsequence[0],
            "constituence_heuristic": results_const[0]
        }
    return {
        "lexical_heuristic": results_lexical,
        "subsequence_heuristic": results_subsequence,
        "constituence_heuristic": results_const
    }


class NliAnalyzer:

    def __init__(self, 
            dataset_name: str, 
            save_dir: str,
            tokenizer_fn,
            tokenizer_name: str,
            subsample_size: int,
            correctness: bool):

        self.save_dir = save_dir
        self.processor = DataProcessor(LOAD_CONFIGS[dataset_name], shuffle=True)
        self.tokenizer = tokenizer_fn
        self.tokenizer_name = tokenizer_name
        self.subsample_size = subsample_size
        self.correctness = correctness

        self.class_names = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    def get_save_affix(self):
        affix = f"_correctness={self.correctness}"
        if self.subsample_size:
            affix += f"_{self.subsample_size}_examples"
        return affix

    def get_sequence_lengths(self, columns, batched=True):
        # Tokenize columns
        token_affix = f"_tokens_{self.tokenizer_name}"
        self.processor.apply_operation(columns, self.tokenizer, batched=batched, affix=token_affix)
        # Calculate Lengths
        tokenized_columns = [column + token_affix for column in columns]
        length_affix = f"_length"
        self.processor.apply_operation(tokenized_columns, get_example_lengths, batched=batched, affix=length_affix)


    def get_n_grams(self, columns, n, batched=True):
        # 1. Tokenize columns of datasets
        token_affix = f"_tokens_{self.tokenizer_name}"
        self.processor.apply_operation(columns, self.tokenizer, batched=batched, affix=token_affix)
        # 2. Calculate n grams
        if n > 1:
            tokenized_columns = [column + token_affix for column in columns]
            gram_affix = f"_{n}_gram"
            self.processor.apply_operation(tokenized_columns, get_example_n_grams, batched=batched, affix=gram_affix, n=n)


    def get_overlaps(self, column_1, column_2, batched=True):
        # Tokenize columns
        token_affix = f"_tokens_{self.tokenizer_name}"
        self.processor.apply_operation([column_1, column_2], self.tokenizer, batched=batched, affix=token_affix)
        
        # Calculate overlap
        tokenized_columns = [column + token_affix for column in [column_1, column_2]]
        overlap_affix = f"_tokens_{self.tokenizer_name}_{column_1}_{column_2}_overlap"
        self.processor.apply_operation(tokenized_columns, 
                                    get_example_overlap, 
                                    batched=batched, 
                                    affix=overlap_affix)

    def get_cos_sim(self, column_1, column_2, model, batched=True):
        self.processor.apply_operation(
            [column_1, column_2], 
            get_example_cosine_sim, 
            batched=batched,
            affix=f"{column_1}_{column_2}_cosine_sim",
            model=model)

    def get_hans_heuristics(self, column_1, column_2, model, batched=True):
        token_affix = f"_tokens_{self.tokenizer_name}"
        self.processor.apply_operation([column_1, column_2], self.tokenizer, batched=batched, affix=token_affix)

        self.processor.apply_operation(
            [column_1 + token_affix, column_2 + token_affix, column_1, column_2],
            get_example_hans_heuristics,
            batched=batched,
            model=model
        )

    def get_yngve_scores(self, columns, model, batched=True):
        self.processor.apply_operation(columns, get_example_yngve, model=model, batched=batched)

    def get_parse_tree_heights(self, columns, model, batched=True):
        self.processor.apply_operation(columns, get_parse_tree_height, model=model, batched=batched)


    def get_sequence_length_stats(self, columns, splits):
        split_data = []
        for split in splits:
            split_data.append(self.processor.get_split_data(split, self.correctness))
        split_data = concatenate_datasets(split_data)

        if self.subsample_size > 0:
            split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)

        for column in columns:
            length_column = column + f"_tokens_{self.tokenizer_name}_length"

            avg = np.mean(split_data[length_column])
            std = np.std(split_data[length_column])

            print(f"column: {column}, avg: {avg}, std: {std}")

    def get_attribute_stats(self, column_1, column_2, splits, attr_column):
        split_data = []
        for split in splits:
            split_data.append(self.processor.get_split_data(split, self.correctness))
        split_data = concatenate_datasets(split_data)

        if self.subsample_size > 0:
            split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)

        #attribute_column = f"_tokens_{self.tokenizer_name}_{column_1}_{column_2}_overlap"

        avg = np.mean(split_data[attr_column])
        std = np.std(split_data[attr_column])

        print(f"avg: {avg}, std: {std}")

    def plot_sequence_lengths(self, columns, splits):
        for column in columns:
            length_column = column + f"_tokens_{self.tokenizer_name}_length"
            plt.figure(figsize=(12, 10))
            texts = []
            for class_ in self.class_names.keys():
                split_data =[]
                for split in splits:
                    split_data.append(self.processor.get_split_data(split, self.correctness))
                #print(split_data)
                
                split_data = concatenate_datasets(split_data)
                if self.subsample_size > 0:
                    split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)

                class_data = split_data.filter(lambda x: x["label"] == class_)
                
                avg = np.mean(class_data[length_column])
                std = np.std(class_data[length_column])
                density = kde.gaussian_kde(class_data[length_column])
                x = list(range(40))
                y = density(x)
                plt.plot(x, y, label=self.class_names[class_])
                plt.axvline(avg)
                texts.append(plt.text(avg + 0.5, 0.01, f"{round(avg, 2)}/{round(std, 3)}"))

            plt.xticks(x, fontsize=8)
            plt.xlabel("length")
            plt.ylabel("density")
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
            plt.legend()
            affix = f"/_{splits}_{column}" + self.get_save_affix()
            plt.savefig(self.save_dir + affix+".png")
            plt.close()

    def plot_attribute_density_per_class(self, splits, column_1, column_2, x_max, attribute_name, attribute_column):
        plt.figure(figsize=(12, 10))
        texts = []
        for class_ in self.class_names.keys():
            split_data = []
            for split in splits:
                split_data.append(self.processor.get_split_data(split, self.correctness))
            split_data = concatenate_datasets(split_data)             
            if self.subsample_size > 0:
                split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)
                
            class_data = split_data.filter(lambda x: x["label"] == class_)
            avg = np.mean(class_data[attribute_column])
            std = np.std(class_data[attribute_column])
            density = kde.gaussian_kde(class_data[attribute_column])
            x = np.linspace(0, 1, 100)
            y = density(x)
            plt.plot(x, y, label=self.class_names[class_])
            plt.axvline(avg)
            texts.append(plt.text(avg+0.01, 0.01, f"{round(avg, 2)}/{round(std, 3)}"))
        
        plt.xlabel(f"{column_1} - {column_2} {attribute_name}")
        plt.ylabel("density")
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
        plt.legend()
        affix = f"/_{splits}_{column_1}_{column_2}_{attribute_name}" + self.get_save_affix()
        plt.savefig(self.save_dir + affix+".png")
        plt.close()

    def get_unique_n_grams(self, splits, columns, n, seed=42):
        #unique_grams = {col: {class_: set() for class_ in self.class_names.keys()} for col in columns}
        split_data = []
            
        for split in splits:
            split_data.append(self.processor.get_split_data(split, self.correctness))
        split_data = concatenate_datasets(split_data)
        if self.subsample_size > 0:
            print("subsampling data")
            split_data = split_data.shuffle(seed=seed)
            split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)
        #class_data = split_data.filter(lambda x: x["label"] == class_)
        unique_grams = set()        
        for column in columns:
            if n == 1:
                gram_column = column + f"_tokens_{self.tokenizer_name}"
            else:
                gram_column = column + f"_tokens_{self.tokenizer_name}_{n}_gram"
            unique_grams.update(
                set(
                    tuple(n_gram) for example in split_data[gram_column]
                    for n_gram in example
                )
            )
        return len(unique_grams)
        #return len(unique_grams) / len(split_data[gram_column])
        #return {column: {class_: len(unique_grams[column][class_]) for class_ in self.class_names.keys()} for column in columns}

    def get_group_similarities(self, group_df, sim_col, model):
        # returns the average max similarity per group
        # NOTE: We are skipping groups with <= 2 hypotheses.
        similarities = []
        for name, group in group_df:
            if len(group[sim_col]) > 1:
                hypothesis_embeddings = model.encode(group[sim_col].values)
                cosine_sims = util.cos_sim(hypothesis_embeddings, hypothesis_embeddings)
                cosine_sims.fill_diagonal_(0)
                max_sims = torch.max(cosine_sims)
                mean_sims = torch.mean(max_sims)
                similarities.append(mean_sims.item())
        return similarities

    def get_group_BLEU_similarities(self, group_df, sim_col_tokens, sim_col):
        similarities = []
        for name, group in group_df:
            if len(group[sim_col]) > 1:
                references = [list(x) for x in group[sim_col_tokens].values]
                scores = torch.zeros((len(references), len(references)))
                for i in range(len(references)):
                    for j in range(len(references)):
                        if i != j:
                            scores[i][j] = nltk.translate.bleu_score.sentence_bleu([references[i]], references[j])
                scores.fill_diagonal_(0)
                max_sims = torch.max(scores)
                mean_sims = torch.mean(max_sims)
                similarities.append(mean_sims.item())
        return similarities

    def get_interBLEU(self, group_col, sim_col, splits):
        token_affix = f"_tokens_{self.tokenizer_name}"
        self.processor.apply_operation([sim_col], self.tokenizer, batched=True, affix=token_affix)

        split_data = []
        for split in splits:
            split_data.append(self.processor.get_split_data(split, self.correctness))
        split_data = concatenate_datasets(split_data)
        
        if self.subsample_size > 0:
            split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)

        similarities = []
        for class_ in self.class_names.keys():
            class_data = split_data.filter(lambda x: x["label"] == class_)
            
            class_data.set_format("pandas")
            #print(len(split_data[group_col].unique()))
            sim_col_tokens = sim_col + f"_tokens_{self.tokenizer_name}"
            similarities.extend(self.get_group_BLEU_similarities(class_data[:].groupby(group_col), sim_col_tokens, sim_col))

        #avg = np.mean(similarities)
        #std = np.std(similarities)
        #print(f"avg: {avg}, std: {std}")

        return similarities

    def get_intercosine(self, group_col, sim_col, splits, model):
        split_data = []
        for split in splits:
            split_data.append(self.processor.get_split_data(split, self.correctness))
        split_data = concatenate_datasets(split_data)
        
        if self.subsample_size > 0:
            split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)
        
        similarities = []
        for class_ in self.class_names.keys():
            class_data = split_data.filter(lambda x: x["label"] == class_)

            class_data.set_format("pandas")
            similarities.extend(self.get_group_similarities(class_data[:].groupby(group_col), sim_col, model))
        return similarities


    def plot_interexample_similarity(self, group_col, sim_col, splits, model):
        class_sims = {l:[] for l in self.class_names.keys()}
        plt.figure(figsize=(12, 10))
        texts = []
        for class_ in self.class_names.keys():
            split_data = []
            for split in splits:
                split_data.append(self.processor.get_split_data(split, self.correctness))
            split_data = concatenate_datasets(split_data)

            if self.subsample_size > 0:
                split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)

            class_data = split_data.filter(lambda x: x["label"] == class_)
            
            class_data.set_format("pandas")
            similarities = self.get_group_similarities(class_data[:].groupby(group_col), sim_col, model)
            density = kde.gaussian_kde(similarities)
            avg = np.mean(similarities)
            std = np.std(similarities)
            x = np.linspace(0, 1, 100)
            y = density(x)
            plt.plot(x, y, label=self.class_names[class_])
            plt.axvline(avg)
            texts.append(plt.text(avg + 0.01, 0.01, f"{round(avg, 2)}/{round(std, 3)}"))
            
        plt.xlabel(f"Cosine Similarity")
        plt.ylabel(f"density")
        plt.legend()
        affix = f"/_{splits}_{group_col}_{sim_col}_interexampe_similarity" + self.get_save_affix()
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
        plt.savefig(self.save_dir + affix + ".png")
        plt.close()

    def plot_hans_heuristics(self, splits):
        positives, negatives = [0, 0, 0], [0, 0, 0]
        labels = ["lexical", "subsequence", "constituence"]
        split_data = []
        for split in splits:
            split_data.append(self.processor.get_split_data(split, self.correctness))
        split_data = concatenate_datasets(split_data)

        if self.subsample_size > 0:
            split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)

        for example in split_data:
            if example["lexical_heuristic"] == 1:
                if example["label"] == 0:
                    positives[0] += 1
                else:
                    negatives[0] += 1

            if example["subsequence_heuristic"] == 1:
                if example["label"] == 0:
                    positives[1] += 1
                else:
                    negatives[1] += 1

            if example["constituence_heuristic"] == 1:
                if example["label"] == 0:
                    positives[2] += 1
                else:
                    negatives[2] += 1
        texts = []
        for i in range(len(positives)):
            total = positives[i] + negatives[i]
            try:
                perc_neg = negatives[i] / total
            except:
                perc_neg = 0
            print(total, perc_neg)
            plt.annotate(f"{total}/{perc_neg}", (i, total + 2))

        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=45)
        plt.bar(labels, positives, width=0.35, label="positive")
        plt.bar(labels, negatives, width=0.35, bottom=positives, label="negative")
        plt.ylabel("Number of examples")
        plt.xlabel("Heuristic type")
        plt.legend()
        #adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
        affix = f"/_{splits}_hans_heuristics" + self.get_save_affix()
        plt.savefig(self.save_dir + affix + ".png")
        plt.close()


    def get_token_label_counts(self, columns, splits, batched=True):
        token_affix = f"_tokens_{self.tokenizer_name}"
        self.processor.apply_operation(columns, self.tokenizer, batched=batched, affix=token_affix)

        counts = defaultdict(lambda: defaultdict(int))
        for column in columns:
            token_column = column + token_affix

            if len(splits) > 0:
                split_data = []
                for split in splits:
                    split_data.append(self.processor.get_split_data(split, self.correctness))
                split_data = concatenate_datasets(split_data)
            else:
                split_data = self.processor.dataset
            
            if self.subsample_size > 0:
                split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)

            for example in tqdm(split_data):
                for token in example[token_column]:
                    if example["label"] >= 0:
                        counts[token][example["label"]] += 1
        return counts

    def get_gram_label_counts(self, n, columns, splits, batched=True):
        token_affix = f"_tokens_{self.tokenizer_name}"
        if n > 1:
            token_affix += f"_{n}_gram"
        self.get_n_grams(columns, n, batched=batched)
        
        counts = defaultdict(lambda: defaultdict(int))
        token_columns = [col + token_affix for col in columns]
        split_data = []
        for split in splits:
            split_data.append(self.processor.get_split_data(split, self.correctness))
        split_data = concatenate_datasets(split_data)

        if self.subsample_size > 0:
            split_data = self.processor.subsample_split_classes(split_data, self.subsample_size)

        for example in tqdm(split_data):
            tokens = [x for y in [example[column] for column in token_columns] for x in y]
            for token in tokens:
                if example["label"] >= 0:
                    counts[" ".join(token)][example["label"]] += 1
        return counts


    def get_z_statistics(self, counts):
        # Hypothesis probability
        p_0 = 1/3

        z_statistics = defaultdict(dict)
        for x, label_counts in counts.items():
            n = sum(c for c in label_counts.values())
            if n >= 20:
                z_statistics[x] = {
                    label: ((count/n) - p_0) * ((p_0*(1-p_0)/n)**-0.5)
                    for label, count in label_counts.items()
                }
        return z_statistics

    def get_emperical_label_probabilities(self, counts):
        label_probabilities = defaultdict(dict)
        for x, label_counts in counts.items():
            n = sum(c for c in label_counts.values())
            if n >= 20:
                label_probabilities[x] = {
                    label: count/n
                    for label, count in label_counts.items()
                }
        return label_probabilities

    def get_z_curve(self, n, correction=0.01/28000):
        # TODO: Does z needs to be 5?
        significance_level = 5 - correction
        p_0 = 1/3
        curve = (significance_level * 2**0.5) / (3 * n**0.5) + p_0
        return curve

    
    def plot_z_statistics(self, columns, splits):
        themes.theme_few(grid=False, fontsize=14)
        n = [1, 2, 3, 4, 5, 6]
        
        fig, axs = plt.subplots(2, 3, sharey=True)
        axs = [x for y in axs for x in y]

        for i in range(len(n)):
            counts = self.get_gram_label_counts(n[i], columns, splits)
            z_statistics = self.get_z_statistics(counts)
            X = {0: [], 1: [], 2: []}
            for x in z_statistics.keys():
                for label, z in z_statistics[x].items():
                    X[label].append(z)

            for label, values in X.items():
                axs[i].hist(values, 30, histtype="step", label=label, density=True, range=(-10, 10))
                axs[i].set_title(f"{n[i]}")

        fig.supxlabel("z-score")
        fig.supylabel("Count")
        fig.legend(["Entailment", "Neutral", "Contradiction"], ncol=3, loc="upper center")
        plt.show()



    def plot_artefacts(self, columns, splits):
        counts = self.get_token_label_counts(columns, splits)
        probs = self.get_emperical_label_probabilities(counts)
        z_stats = self.get_z_statistics(counts)
        
        plt.figure(figsize=(12, 10))
 
        X = []

        for x_idx, (x, label_probs) in enumerate(probs.items()):
            n = sum(c for c in counts[x].values())
            for label, prob in label_probs.items():
                X.append([label, prob, n, x_idx, z_stats[x][label]])
        
        X = np.array(X)[np.array(X)[:,-1].argsort()][::-1]
        z_curve = self.get_z_curve(X[:, 2])
        artefact_idx = np.argwhere(z_curve < X[:, 1])[:, 0]
        normal_idx = np.argwhere(z_curve >= X[:, 1])[:, 0]
        # sort artefacts by z-score descending.
        artefacts = X[artefact_idx]
        # select points to annotate
        annotations = X[:10]
        annotation_texts = np.array(list(probs.keys()))[annotations[:,3].astype(int)]
        
        plt.scatter(X[:, 2][normal_idx], X[:, 1][normal_idx], s=2, color="grey", alpha=0.3)
        plt.plot(np.sort(X[:, 2]), self.get_z_curve(np.sort(X[:, 2])), label=r"$\alpha=0.01/28k$")

        label_0 = np.argwhere(artefacts[:,0] == 0)[:,0]
        label_1 = np.argwhere(artefacts[:,0] == 1)[:,0]
        label_2 = np.argwhere(artefacts[:,0] == 2)[:,0]
        #print(label_0)
        #print(len(label_0), len(label_1), len(label_2))
        print(len(label_0), len(label_1), len(label_2), len(label_0) + len(label_1) + len(label_2)) 
        #plt.scatter(artefacts[:,2][label_0], 
        #            artefacts[:,1][label_0], 
        #            label=self.class_names[0], 
        #            color="green", 
        #            s=2, 
        #            alpha=0.5, 
        #            marker='X')
        #plt.scatter(artefacts[:,2][label_1], 
        #            artefacts[:,1][label_1], 
        #            label=self.class_names[1], 
        #            color="orange", 
        #            s=2, 
        #            alpha=0.5)
        #plt.scatter(artefacts[:,2][label_2], 
        #            artefacts[:,1][label_2], 
        #            label=self.class_names[2], 
        #            color="red", 
        #            s=2, 
        #            alpha=0.5, 
        #            marker='s') 
        
        #texts = []
        #for text, data in zip(annotation_texts, annotations):
        #    affix = "$^{}$".format(self.class_names[data[0]][:1])
        #    texts.append(plt.text(data[2], data[1], text + affix))

        #adjust_text(texts)
        #plt.xscale("log")
        #plt.xlabel("n")
        #plt.ylabel(r"$\hat{p}(y|x_i)$")
        #plt.legend()
        #save_loc = f"{self.save_dir}/artefacts_{splits}" + self.get_save_affix()
        #plt.savefig(save_loc + ".png", dpi=600)
        #plt.close()


    def export_pair_ids(self, splits):
        pair_ids = []
        for split in splits:
            pair_ids.extend(self.processor.dataset[split]["pairID"])
        
        pair_ids = list(set(pair_ids))
        with open("pair_ids.pkl", "wb") as f:
            pickle.dump(pair_ids, f)

if __name__ == "__main__":
    datasets = ["snli"]
    columns = ["hypothesis", "premise"]
    analyze_hans_heuristics(datasets, "premise", "hypothesis", 0)
