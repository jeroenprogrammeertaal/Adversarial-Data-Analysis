import os, argparse, random
import wandb
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from train_utils import Logger, DataProcessor
from typing import Union
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast
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
        return model, tokenizer
    if name == "roberta":
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        return model, tokenizer
    if name == "roberta_large":
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        return model, tokenizer

class Trainer:

    def __init__(self, 
        model,
        dataprocessor,
        logger,
        config:dict,
    ):
        
        self.model = model.to(config["device"])
        #self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config["gpu"]])
        self.data_processor = dataprocessor
        self.logger = logger
        self.config = config
        self.optimizer = self.init_optimizer()
        self.grad_scaler = GradScaler()
        self.lr_scheduler = self.init_lr_scheduler()
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config["gpu"]])
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

    def pre_allocate_memory(self):
        random_batch = self.data_processor.generate_random_batch()
        self.model.train()
        with autocast():
            outputs = self.model(**random_batch)
        loss = outputs.loss
        loss.backward()
        self.optimizer.zero_grad(set_to_none=True)

    def log(self, metrics):
        if self.logger is None:
            return
        self.logger.log(metrics)

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        batch = self.to_device(batch)
        
        with autocast():
            outputs = self.model(**batch)

        loss = outputs.loss
        logits = outputs.logits.detach()
        accuracy = self.get_accuracy(logits, batch["labels"].detach())
        
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.lr_scheduler.step()
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        return {
            "train/loss": loss.detach().item(), 
            "train/accuracy": accuracy,
            "train/learning_rate": self.get_lr(),
            "train/grad_scale": scale
        }

    def train(self):
        wandb.watch(self.model, log="all", log_freq=10)
        train_loader = self.data_processor.get_dataloader("train")
        validation_loader = self.data_processor.get_dataloader("validation")
        test_loader = self.data_processor.get_dataloader("test")
        progress_bar = tqdm(range(self.get_n_training_steps()))

        if self.config["pre_allocate_memory"]:
            self.pre_allocate_memory()
        
        for epoch in range(self.config["epochs"]):
            self.model.train()
            
            for step, batch in enumerate(train_loader):
                step_metrics = self.train_step(batch)
                self.log(step_metrics)
                progress_bar.update(1)

            if validation_loader:
                validation_metrics = self.validate(validation_loader, "validation", epoch)
                self.log(validation_metrics)

            if test_loader:
                test_metrics = self.validate(test_loader, "test", epoch)
                self.log(test_metrics)
            
            # Gather predictions for dataset cartography
            self.validate(train_loader, "train", epoch)

        self.export_weights(epoch)
        self.logger.log_cartography()

    def validation_step(self, batch, split, epoch):
        idx = batch["idx"]
        batch = self.to_device(batch)
        
        with autocast():
            outputs = self.model(**batch)

        loss = outputs.loss
        logits = outputs.logits.detach()
        
        labels = batch["labels"].detach()
        accuracy = self.get_accuracy(logits, labels)
        correctness = (torch.argmax(logits, dim=-1) == labels).int()
        label_probabilities = self.get_label_probabilities(logits, labels)
        self.logger.store_label_probabilities(idx, label_probabilities, correctness, split, epoch)
        
        return {
            "loss": loss.detach().item(), 
            "accuracy": accuracy
        }
    
    @torch.no_grad()
    def validate(self, loader, split:str, epoch):
        self.model.eval()
        n_batches = len(loader)
        avg_accuracy = 0
        avg_loss = 0
        
        for batch in loader:
            step_metrics = self.validation_step(batch, split, epoch)
            avg_loss += step_metrics["loss"] / n_batches
            avg_accuracy += step_metrics["accuracy"] / n_batches
        
        return {
            "epoch": epoch,
            f"{split}/loss": avg_loss, 
            f"{split}/accuracy": avg_accuracy
        }

    def export_weights(self, epoch):
        if self.config["main"]:
            self.model = self.model.to(torch.device("cpu"))
            path = self.logger.get_save_affix() + ".pt"
            torch.save(self.model.module.state_dict(), path)

def train(gpu, args):
    args['gpu'] = gpu # local rank
    args['rank'] = args["node_rank"] * args["gpus"] + gpu # global rank

    dist.init_process_group(backend='nccl', init_method="env://", world_size=args["world_size"], rank=args["rank"])

    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])

    torch.cuda.device(args['gpu'])
    args["main"] = (args["rank"] == 0)
    args["device"] = args["gpu"]

    # Init data processor
    model, tokenizer = prepare_model(args["model_name"])
    dataprocessor = DataProcessor(args["dataset_name"], tokenizer, args)
    split_sizes = {split: dataprocessor.get_split_n_examples(split) for split in args["splits"]}
    logger = Logger(args, split_sizes)
    trainer = Trainer(model, dataprocessor, logger, args)
    trainer.train()
    wandb.finish()

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
    parser.add_argument("--pre_allocate_memory", action="store_true",
                        help="Whether to run dummy batches before training to pre-allocate memory.")
    parser.add_argument("--empty_input", action="store_true",
                        help="Whether to train on empty inputs.")

    # Device
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use.")
    parser.add_argument("--nodes", type=int, default=1,
                        help="Number of nodes to use.")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="Ranking of node to use.")
    args = vars(parser.parse_args())
    args["world_size"] = args["gpus"] * args["nodes"]
    args["dist_url"] = f"tcp://localhost:8888"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(args["gpus"])])
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = '12355'
    print(f"Found {torch.cuda.device_count()} gpus...")
        
    args["splits"] = ["train"]
    if args["validation_splits"] != ["none"]:
        args["splits"].append("validation")
    if args["test_splits"] != ["none"]:
        args["splits"].append("test")
   
    mp.spawn(train, args=(args,), nprocs=args['gpus'])
    model, tokenizer = prepare_model("tiny_bert")
