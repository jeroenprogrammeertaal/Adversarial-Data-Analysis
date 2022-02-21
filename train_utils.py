import os
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from data_utils.load_configs import LOAD_CONFIGS
from data_utils.load_data import get_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import concatenate_datasets, DatasetDict

class Logger:

    def __init__(self, train_config, split_sizes):
        self.config = train_config
        self.best_validation_loss = float("inf")
        self.label_probabilities = {
            "label_probs": {
                split: torch.zeros(split_sizes[split], self.config["epochs"], device=self.config["device"], dtype=torch.float16)
                for split in self.config["splits"]
            },
            "correctness": {
                split: torch.zeros(split_sizes[split], device=self.config["device"], dtype=torch.float16)
                for split in self.config["splits"]
            }
        }
    
        self.init_wandb()

    def init_wandb(self):
        os.environ["WANDB_API_KEY"] = ""
    
        wandb.init(
            project="adversarial_NLU",
            group=f"{self.config['model_name']}_{self.config['dataset_name']}",
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

    def store_label_probabilities(self, idx, label_probabilities, correctness, split, epoch):
        self.label_probabilities["label_probs"][split][idx, epoch] = label_probabilities.squeeze()
        self.label_probabilities["correctness"][split][idx] += correctness 


    def log_histogram(self, data, metric, split):
        plt.hist(data, bins=10, edgecolor="white")
        plt.xlabel(metric)
        plt.ylabel("Number of examples")
        wandb.log({f"{split}/{metric}_histogram": wandb.Image(plt)})
        plt.close()

    def log_scatter(self, data, split):
        sns.scatterplot(data[:, 1], data[:, 0], hue=data[:, 2], hue_norm=(0, 1))
        plt.xlabel("Variability")
        plt.ylabel("Confidence")
        wandb.log({f"{split}/variability_confidence_plot": wandb.Image(plt)})
        plt.close()

    def log_cartography(self):
        for split in self.config["splits"]:
            data = torch.stack((
                torch.mean(self.label_probabilities["label_probs"][split], dim=-1),
                torch.std(self.label_probabilities["label_probs"][split], dim=-1),
                self.label_probabilities["correctness"][split] / self.config["epochs"]
            ), dim=1).cpu().numpy()

            self.log_histogram(data[:,0], "confidence", split)
            self.log_histogram(data[:,1], "variability", split)
            self.log_histogram(data[:,2], "correctness", split)
            
            self.log_scatter(data, split)
            self.export_cartography(data, split)


    def get_save_affix(self):
        path = f"{self.config['save_dir']}/{self.config['dataset_name']}/"
        path += f"{self.config['model_name']}_{self.config['train_splits']}_"
        path += f"epoch={self.config['epochs']}_seed={self.config['seed']}_"
        path += f"batchsize={self.config['batch_size']}_betas={self.config['betas']}"
        path += f"lr={self.config['lr']}_weight_decay={self.config['weight_decay']}_"
        path += f"warmup_steps={self.config['warmup_ratio']}"
        if self.config["empty_input"]:
            path += "_empty_input=True"
        return path

    def export_cartography(self, data, split):
        path = self.get_save_affix() + f"_{split}_cartography_rank={self.config['rank']}.pt"
        torch.save(data, path)
    


class DataProcessor:

    def __init__(self, dataset_name, tokenizer, train_config):
        self.data_config = LOAD_CONFIGS[dataset_name]
        self.data_config["cache_dir"] = train_config["dataset_cache_dir"]
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.batch_columns = ["idx"] + self.tokenizer.model_input_names + ["label"]
        self.label_names = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
        self.dataset = self.prepare_dataset()
        self.dataloaders = self.prepare_dataloaders()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def tokenize_func(self, examples):
        return self.tokenizer(
            examples["premise"],
            examples["hypothesis"],
            add_special_tokens=True,
            padding="longest",
            truncation="only_first"
        )

    def prepare_dataset_splits(self, splits):
        dataset = get_dataset(self.data_config["dataset"], **self.data_config)
        dataset = self.get_splits(dataset, splits)
        dataset = dataset.shuffle(self.train_config["seed"])
        if "gold" in dataset.column_names:
            dataset = dataset.rename_column("gold", "label")
            dataset = dataset.map(lambda x: {"label": self.label_names[x["label"]]})

        dataset = dataset.filter(lambda x: 0 <= x["label"] < 3)

        if self.train_config["empty_input"]:
            dataset = dataset.map(lambda x: {"premise":"", "hypothesis":""})

        tokenized_dataset = (
            dataset.map(self.tokenize_func, batched=True, batch_size=self.train_config["batch_size"])
            .map(lambda _, idx: {'idx':idx}, with_indices=True)
            .remove_columns([col for col in dataset.column_names if col not in self.batch_columns])
            .rename_column("label", "labels")
        )
        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def prepare_dataset(self):
        data_dict = {}
        data_dict["train"] = self.prepare_dataset_splits(self.train_config["train_splits"])
        if "validation" in self.train_config["splits"]:
            data_dict["validation"] = self.prepare_dataset_splits(self.train_config["validation_splits"])
        if "test" in self.train_config["splits"]:
            data_dict["test"] = self.prepare_dataset_splits(self.train_config["test_splits"])
        return DatasetDict(data_dict)

    def get_splits(self, dataset, splits):
        if splits is None:
            return
        return concatenate_datasets([dataset[split] for split in splits])

    def init_dataloader(self, dataset):
        return DataLoader(dataset, shuffle=False, batch_size=self.train_config["batch_size"], num_workers=4, pin_memory=True)

    def init_ddp_dataloader(self, dataset):
        sampler = DistributedSampler(
            dataset, 
            shuffle=False, 
            num_replicas=self.train_config["world_size"],
            rank=self.train_config["rank"]
        )
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.train_config["batch_size"],
            num_workers=4,
            pin_memory=True
        )

    def prepare_dataloaders(self):
        if not "gpus" in self.train_config.keys():
            return {
                split: self.init_dataloader(self.dataset[split]) 
                for split in self.train_config["splits"]
            }
        return {
            split: self.init_ddp_dataloader(self.dataset[split])
            for split in self.train_config["splits"]
        }

    def generate_random_batch(self, sequence_length=512):
        random_batch = {}
        random_batch["input_ids"] = torch.randint(0, 10000, (self.train_config["batch_size"], sequence_length), device=self.train_config["device"])
        if "token_type_ids" in self.tokenizer.model_input_names:
            random_batch["token_type_ids"] = torch.zeros(self.train_config["batch_size"], sequence_length, dtype=torch.long, device=self.train_config["device"])
        if "attention_mask" in self.tokenizer.model_input_names:
            random_batch["attention_mask"] = torch.ones(self.train_config["batch_size"], sequence_length, dtype=torch.long, device=self.train_config["device"])
        random_batch["labels"] = torch.zeros(self.train_config["batch_size"], dtype=torch.long, device=self.train_config["device"])
        return random_batch

    def get_dataloader(self, split:str):
        if split in self.dataloaders.keys():
            return self.dataloaders[split]
        return None

    def get_split_size(self, split:str):
        return len(self.dataloaders[split])

    def get_split_n_examples(self, split:str):
        return len(self.dataset[split])




