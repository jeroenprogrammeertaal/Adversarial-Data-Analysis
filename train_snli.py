import os
import argparse
import torch

from train_utils import Logger, DataProcessor
from typing import Union
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def prepare_model(name:str):
    """returns model + tokenizer"""
    if name == "tiny_bert":
        model = AutoModelForSequenceClassification.from_pretrained("google/bert_uncased_L-2_H-128_A-2", num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer
    if name == "bert":
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if name == "roberta":
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

class Trainer:

    def __init__(self, 
        model, 
        data_processor:DataProcessor,
        logger:Logger,
        config:dict,
    ):
        
        self.set_seed(config["seed"])
        self.model = model.to(config["device"])
        self.data_processor = data_processor
        self.logger = logger
        self.config = config
        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()
        self.device = config["device"]

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def to_device(self, batch: dict, with_idx:bool=False):
        if with_idx:
            return {k: v.to(self.device) for k, v in batch.items()}
        return {k: v.to(self.device) for k, v in batch.items() if "idx" not in k}

    def get_n_training_steps(self):
        return self.data_processor.get_split_size("train") * self.config["epochs"]

    def init_optimizer(self):
        # Currently only supports AdamW.
        return AdamW(
            self.model.parameters(),
            lr = self.config["lr"],
            betas = self.config["betas"],
            weight_decay = self.config["weight_decay"]
        )

    def init_lr_scheduler(self):
        training_steps = self.get_n_training_steps()
        warmup_steps = training_steps * self.config["warmup_ratio"]
        if self.config["lr_scheduler"] == "None":
            return get_constant_schedule(self.optimizer)
        if self.config["lr_scheduler"] == "const_warmup":
            return get_constant_schedule_with_warmup(self.optimizer, warmup_steps)
        if self.config["lr_scheduler"] == "cosine":
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, training_steps)
        if self.config["lr_scheduler"] == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, training_steps)
    
    @torch.no_grad()
    def get_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=-1)
        accuracy = torch.sum(predictions == labels) / len(labels)
        return accuracy.item()

    @torch.no_grad()
    def get_label_probabilities(self, logits, labels):
        probabilities = torch.softmax(logits, dim=-1)
        return torch.gather(probabilities, -1, labels[:,None])

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def train_step(self, batch):
        batch = self.to_device(batch)
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        accuracy = self.get_accuracy(logits, batch["labels"])
        loss.backward()

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return {
            "train/loss": loss.item(), 
            "train/accuracy": accuracy,
            "train/learning_rate": self.get_lr()
        }

    def train(self):
        train_loader = self.data_processor.get_dataloader("train")
        validation_loader = self.data_processor.get_dataloader("validation")
        test_loader = self.data_processor.get_dataloader("test")
        progress_bar = tqdm(range(self.get_n_training_steps()))
        
        for epoch in range(self.config["epochs"]):
            self.model.train()
            
            for step, batch in enumerate(train_loader):
                step_metrics = self.train_step(batch)
                self.logger.log(step_metrics)
                progress_bar.update(1)

            validation_metrics = self.validate(validation_loader, "validation", epoch)
            self.logger.log(validation_metrics)

            test_metrics = self.validate(test_loader, "test", epoch)
            self.logger.log(test_metrics)
            
            # Gather predictions for dataset cartography
            self.validate(train_loader, "train", epoch)
            self.validate(test_loader, "test", epoch)

        self.export_weights(epoch)
        self.logger.log_cartography()

    def validation_step(self, batch, split):
        idx = batch["idx"]
        batch = self.to_device(batch)
        
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        
        accuracy = self.get_accuracy(logits, batch["labels"])
        correctness = (torch.argmax(logits, dim=-1) == batch["labels"]).int()
        label_probabilities = self.get_label_probabilities(logits, batch["labels"]).detach()
        self.logger.store_label_probabilities(idx, label_probabilities, correctness, split)
        
        return {
            "loss": loss.item(), 
            "accuracy": accuracy
        }
    
    @torch.no_grad()
    def validate(self, loader, split:str, epoch):
        self.model.eval()
        n_batches = len(loader)
        avg_accuracy = 0
        avg_loss = 0
        
        for batch in loader:
            step_metrics = self.validation_step(batch, split)
            avg_loss += step_metrics["loss"] / n_batches
            avg_accuracy += step_metrics["accuracy"] / n_batches
        
        return {
            "epoch": epoch,
            f"{split}/loss": avg_loss, 
            f"{split}/accuracy": avg_accuracy
        }

    def export_weights(self, epoch):
        self.model = self.model.to(torch.device("cpu"))
        path = self.logger.get_save_affix() + ".pt"
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NLI Classification")
    # Model and Dataset names
    parser.add_argument("--model_name", type=str, default="tiny_bert",
                        help="Name of model to train. Choose from: 'test'.")
    parser.add_argument("--dataset_name", type=str, default="snli",
                        help="Name of dataset to train on.")
    parser.add_argument("--dataset_cache_dir", type=str, default="data",
                        help="Cache directory of data.")
    parser.add_argument("--train_splits", type=str, nargs='+', default=["train"],
                        help="Name(s) of splits to use for training.")
    parser.add_argument("--validation_splits", type=str, nargs='+', default=["validation"],
                        help="Name(s) of splits to use for validation.")
    parser.add_argument("--test_splits", type=str, nargs='+', default=["test"],
                        help="Name(s) of splits to use for testing.")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Path of directory to save model to.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Maximum number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Size of batches.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--lr_scheduler", type=str, default="None",
                        help="Name of lr scheduler to use, choose from: 'None', 'const_warmup', 'cosine'.")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999),
                        help="Adams betas parameters.")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="decoupled weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                        help="Fraction of total training step to apply warmup")

    # Device
    parser.add_argument("--device", default=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    
    args = vars(parser.parse_args())
    
    model, tokenizer = prepare_model(args["model_name"])
    dataprocessor = DataProcessor(args["dataset_name"], tokenizer, args)
    logger = Logger(args)
    trainer = Trainer(model, dataprocessor, logger, args)
    trainer.train()
