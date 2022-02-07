import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from collections import defaultdict
from data_utils.load_configs import LOAD_CONFIGS
from data_utils.load_data import get_dataset
from torch.utils.data import DataLoader
from datasets import concatenate_datasets

class Logger:

    def __init__(self, train_config):
        self.config = train_config
        self.best_validation_loss = float("inf")
        self.label_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        self.init_wandb()

    def init_wandb(self):
        os.environ["WANDB_API_KEY"] = ""

        wandb.init(
            project="adversarial_NLU",
            group=self.config["dataset_name"],
            tags=[self.config["model_name"]],
            config=self.config
        )

        # One validation step will be 1 epoch over validation data.
        wandb.define_metric("epoch")
        wandb.define_metric("validation/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

    def log(self, metrics: dict):
        wandb.log(metrics)
        if "validation/loss" in metrics.keys():
            self.update_best_loss(metrics)

    def update_best_loss(self, metrics):
        if self.best_validation_loss > metrics["validation/loss"]:
            self.best_validation_loss = metrics["validation/loss"]

    def store_label_probabilities(self, idx, label_probabilities, correctness, split):
        for idx, label_prob, correct in zip(idx, label_probabilities, correctness):
            self.label_probabilities[split][idx.item()]["label_prob"].append(label_prob.item())
            self.label_probabilities[split][idx.item()]["correctness"].append(correct.item())


    def log_histogram(self, data, metric, split):
        # TODO: log to wandb
        plt.hist(data, bins=10, edgecolor="white")
        plt.xlabel(metric)
        plt.ylabel("Number of examples")
        wandb.log({f"{split}/{metric}_histogram": wandb.Image(plt)})
        plt.close()

    def log_scatter(self, data, split):
        sns.scatterplot(data[:, 2], data[:, 1], hue=data[:,3], hue_norm=(0, 1))
        plt.xlabel("Variability")
        plt.ylabel("Confidence")
        
        wandb.log({f"{split}/variability_confidence_plot": wandb.Image(plt)})
        plt.close()

    def log_cartography(self):
        for split in self.label_probabilities.keys():
            # Calculate confidences and variabilities
            data = np.array([
                [x,
                np.mean(self.label_probabilities[split][x]["label_prob"]),
                np.std(self.label_probabilities[split][x]["label_prob"]),
                np.sum(self.label_probabilities[split][x]["correctness"]) / len(self.label_probabilities[split][x]["correctness"])] 
                for x in self.label_probabilities[split].keys()
            ])

            self.log_histogram(data[:,1], "confidence", split)
            self.log_histogram(data[:,2], "variability", split)
            self.log_histogram(data[:,3], "correctness", split)
            self.log_scatter(data, split)
            self.export_cartography(data, split)


    def get_save_affix(self):
        path = f"{self.config['save_dir']}/{self.config['dataset_name']}/"
        path += f"{self.config['model_name']}_{self.config['train_splits']}_"
        path += f"epoch={self.config['epochs']}_seed={self.config['seed']}_"
        path += f"batchsize={self.config['batch_size']}_betas={self.config['betas']}"
        path += f"lr={self.config['lr']}_weight_decay={self.config['weight_decay']}_"
        path += f"warmup_steps={self.config['warmup_ratio']}"
        return path

    def export_cartography(self, data, split):
        path = self.get_save_affix() + f"_{split}_cartography.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
    


class DataProcessor:

    def __init__(self, dataset_name, tokenizer, train_config):
        self.data_config = LOAD_CONFIGS[dataset_name]
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.dataset = self.prepare_dataset()
        self.dataloaders = self.prepare_dataloaders()

    def tokenize_func(self, examples):
        return self.tokenizer(
            examples["premise"],
            examples["hypothesis"],
            add_special_tokens=True,
            padding="max_length",
            truncation="only_first"
        )

    def prepare_dataset(self):
        dataset = get_dataset(self.data_config["dataset"], **self.data_config)

        tokenized_dataset = (
            dataset.map(self.tokenize_func, batched=True)
            .map(lambda _, idx: {'idx':idx}, with_indices=True)
            .remove_columns(["premise", "hypothesis"])
            .rename_column("label", "labels")
        )
        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def get_splits(self, splits):
        if splits is None:
            return
        return concatenate_datasets([self.dataset[split] for split in splits])

    def init_dataloader(self, dataset):
        return DataLoader(dataset, shuffle=True, batch_size=self.train_config["batch_size"])

    def prepare_dataloaders(self):
        return {
            "train": self.init_dataloader(self.get_splits(self.train_config["train_splits"])),
            "validation": self.init_dataloader(self.get_splits(self.train_config["validation_splits"])),
            "test": self.init_dataloader(self.get_splits(self.train_config["test_splits"]))
        }

    def get_dataloader(self, split:str):
        return self.dataloaders[split]

    def get_split_size(self, split:str):
        return len(self.dataloaders[split])






