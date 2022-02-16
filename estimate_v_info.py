import argparse
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from train_utils import DataProcessor
from train_snli import prepare_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_model_path(args, empty_input=False):
    path = f"{args['save_dir']}/{args['dataset_name']}/"
    path += f"{args['model_name']}_{args['train_splits']}_"
    path += f"epoch={args['epochs']}_seed={args['seed']}_"
    path += f"batchsize={args['batch_size']}_betas={args['betas']}"
    path += f"lr={args['lr']}_weight_decay={args['weight_decay']}_"
    path += f"warmup_steps={args['warmup_ratio']}"
    if empty_input:
        path += "_empty_input=True"
    path += ".pt"
    return path

def load_model(args):
    model, tokenizer = prepare_model(args["model_name"])
    model_path = get_model_path(args)
    empty_model_path = get_model_path(args, empty_input=True)

    empty_model = deepcopy(model)
    model.load_state_dict(torch.load(model_path))
    empty_model.load_state_dict(torch.load(empty_model_path))
    
    return model, empty_model, tokenizer

def construct_empty_batch(tokenizer, batch_size):
    return tokenizer([""] * batch_size, return_tensors='pt')

def batch_to_device(batch, device, with_idx:bool=False):
    if with_idx:
        return {k:v.to(device) for k, v in batch.items()}
    return {k:v.to(device) for k, v in batch.items() if "idx" not in k}

def get_label_probabilities(model, batch, labels):
    logits = model(**batch).logits
    correctness = (torch.argmax(logits, dim=-1) == labels).float()
    probs = torch.softmax(logits, dim=-1)
    return torch.gather(probs, -1, labels[:, None]), correctness

@torch.no_grad()
def estimate_pvi(g, g_prime, data_loader, tokenizer, n, args):
    g = g.to(args["device"])
    g_prime = g_prime.to(args["device"])

    # [pvi, correctness]
    pvi = torch.zeros(n, 2, device=args["device"])
    
    g.eval()
    g_prime.eval()
    for batch in data_loader:
        idx = batch["idx"]
        batch = batch_to_device(batch, args["device"])
        bsz = batch["input_ids"].size(0)

        empty_batch = construct_empty_batch(tokenizer, bsz)
        empty_batch = batch_to_device(empty_batch, args["device"])

        g_probs, _ = get_label_probabilities(g, empty_batch, batch["labels"])
        g_prime_probs, g_prime_correct = get_label_probabilities(g_prime, batch, batch["labels"])

        pvi[idx,0] = torch.squeeze(-torch.log2(g_probs) + torch.log2(g_prime_probs))
        pvi[idx,1] = g_prime_correct 

    return pvi

def plot_pvi(pvi, split):
    correct = (pvi[:, 1] == 1)
    plt.hist(pvi[correct,0], bins=20, edgecolor="white", range=(-2,2), alpha=0.4, label="correct")
    plt.hist(pvi[~correct, 0], bins=20, edgecolor="white", range=(-2,2), alpha=0.4, label="incorrect")
    plt.xlabel("pvi")
    plt.ylabel("Number of examples")
    plt.legend()
    path = get_model_path(args).replace(".pt", f"_pvi_{split}.png")
    plt.savefig(path)
    plt.close()


def v_info_experiment(g, g_prime, dataprocessor, args):
    
    split_sizes = {
        "train": sum([dataprocessor.get_split_n_examples(split) for split in args['train_splits']]),
        "validation": sum([dataprocessor.get_split_n_examples(split) for split in args['validation_splits']]),
        "test": sum([dataprocessor.get_split_n_examples(split) for split in args['test_splits']])
    }

    for split in ["train", "validation", "test"]:
        loader = dataprocessor.get_dataloader(split)
        pvi = estimate_pvi(g, g_prime, loader, dataprocessor.tokenizer, split_sizes[split], args)
        
        v_info = torch.mean(pvi)
        print(f"V Information in {split} set: {v_info.item()}")
        path = get_model_path(args).replace(".pt", f"_pvi_{split}.pt")
        pvi = pvi.cpu()
        torch.save(pvi, path)
        plot_pvi(pvi.numpy(), split)

        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate V-information")

    parser.add_argument("--model_name", type=str, default="tiny_bert",
                        help="Name of model architecture. Choose from: tiny_bert, bert, roberta.")
    parser.add_argument("--dataset_name", type=str, default="snli",
                        help="Name of dataset to use.")
    parser.add_argument("--dataset_cache_dir", type=str, default="data",
                        help="Location of dataset cache directory.")
    parser.add_argument("--train_splits", type=str, nargs="+", default=["train"],
                        help="Names of splits used for training.")
    parser.add_argument("--validation_splits", type=str, nargs="+", default=["validation"],
                        help="Names of splits used for validation.")
    parser.add_argument("--test_splits", type=str, nargs="+", default=["test"],
                        help="Names of splits used for testing.")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed used for training model")

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs used to train model.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch Size used during training.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate used during training.")
    parser.add_argument("--lr_scheduler", type=str, default="None",
                        help="Name of lr scheduler used during training.")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999),
                        help="Adams betas parameters.")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="decoupled_weight decay value.")
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                        help="Ratio of warmup steps.")
    parser.add_argument("--device", default=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    args = vars(parser.parse_args())
    args["empty_input"] = False

    args["splits"] = ["train"]
    if args["validation_splits"] != ["none"]:
        args["splits"].append("validation")
    if args["test_splits"] != ["none"]:
        args["splits"].append("test")

    model, empty_model, tokenizer = load_model(args)
    dataprocessor = DataProcessor(args["dataset_name"], tokenizer, args)

    v_info_experiment(empty_model, model, dataprocessor, args)
