"""Code slightly adjusted from: https://github.com/m-hahn/sensitivity/blob/main/code/xlnet/GLUE/MNLI/generate18_c.py
"""
import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from construct_subset_dataset import get_batches

PAD_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

def load_model():
    #tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-xlnet-base-cased")
    #model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-xlnet-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    model = AutoModelForCausalLM.from_pretrained("xlnet-base-cased")
    return model, tokenizer


def get_first_mask_idx(subsets_ids):
    mask_tokens = (subsets_ids == 6).int()
    idx = torch.arange(mask_tokens.size(1), 0, -1, device=subsets_ids.device)
    first_indices = torch.argmax(mask_tokens * idx, 1, keepdim=True)
    return first_indices

def construct_permutation_mask(subsets_ids):
    permutation_mask = torch.zeros((subsets_ids.size(0), subsets_ids.size(1), subsets_ids.size(1)), dtype=torch.float, device=subsets_ids.device)
    mask_idx = torch.nonzero((subsets_ids == 6).int())
    permutation_mask[mask_idx[:,0], :, mask_idx[:,1]] = 1.0
    return permutation_mask

def construct_single_target_mapping(subsets_ids):
    first_target_idx = get_first_mask_idx(subsets_ids)
    target_mapping = torch.zeros((subsets_ids.size(0), subsets_ids.size(1)), dtype=torch.float, device=subsets_ids.device)
    target_mapping.scatter_(1, first_target_idx, 1.0)
    return target_mapping.unsqueeze(1)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.determininistic = True
    #torch.backends.cudnn.benchmark = False

def get_dataloader(tokenizer, rank, args):
    batches = get_batches(tokenizer, args)
    part_size = len(batches) // args["world_size"]
    part_rank = int(part_size * rank)
    start = int(part_rank * part_size)
    end = int((part_rank + 1) * part_size)
    if end > len(batches):
        end = len(batches)
    return batches[start : end]

def to_device(batch, gpu):
    return [x.cuda(gpu) for x in batch]

@torch.no_grad()
def generate_token(model, batch, vocab_mask, gpu):
    # Construct inputs
    input_ids, attn_mask = batch
    permutation_mask = construct_permutation_mask(input_ids)
    target_mask = construct_single_target_mapping(input_ids)
    input_ids, attn_mask, permutation_mask, target_mask = to_device([input_ids, attn_mask, permutation_mask, target_mask], gpu)
    
    # Generate new tokens
    outputs = model(
        input_ids,
        attention_mask=attn_mask,
        perm_mask=permutation_mask,
        target_mapping=target_mask
    )
    probs = F.softmax(outputs.logits + vocab_mask, dim=-1).squeeze(1)
    next_tokens = torch.multinomial(probs, num_samples=1)

    # Insert generated tokens
    target_idx = get_first_mask_idx(input_ids)
    input_ids.scatter_(1, target_idx, next_tokens)

    return input_ids.detach()

def tokens_to_str(tokens, tokenizer):
    strings = []
    for sequence in tokens:
        strings.append(tokenizer.decode(sequence, skip_special_tokens=False).replace("<pad>", ""))
    return strings

@torch.no_grad()
def generate_neighbours(gpu, args):
    rank = args["node_rank"] * args["gpus"] + gpu
    dist.init_process_group(
        backend="nccl",
        init_method=args["dist_url"],
        world_size=args["world_size"],
        rank=rank
    )
    set_seed(args["seed"])

    torch.cuda.set_device(gpu)

    model, tokenizer = load_model()
    vocab_mask = torch.cuda.FloatTensor([float("-inf") if ("<" in tokenizer.convert_ids_to_tokens(x)) else 0 for x in range(32000)]).view(1, 1, -1)
    pad_length = len(tokenizer(PAD_TEXT, add_special_tokens=False, return_tensors="pt")["input_ids"][0])

    results = []
    model = model.cuda(gpu)
    model.eval()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    data_loader = get_dataloader(tokenizer, rank, args)
    for batch in data_loader:
        for _ in range(args["n_neighbour_samples"]):
            input_ids, attn_mask, example_ids, subset_masks, sep_idc = batch
            n_remaining_targets = torch.sum((input_ids[0] == 6).int())
            n_targets = n_remaining_targets.item()

            while n_remaining_targets > 0:
                input_ids = generate_token(model, (input_ids, attn_mask), vocab_mask, gpu)
                n_remaining_targets = torch.sum((input_ids[0] == 6).int())

            for i in range(input_ids.size(0)):
                sep_idx = sep_idc[i].item() + pad_length
                decoded = tokens_to_str([input_ids[i, pad_length:sep_idx], input_ids[i, sep_idx:]], tokenizer)
                results.append((decoded[0], decoded[1], n_targets, example_ids[i].item(), subset_masks[i]))
    
        with open(args["save_dir"] + f"/{args['dataset_name']}/_n_samples={args['n_samples']}_split={args['split']}_seed={args['seed']}_rank={rank}.txt", "ab") as f:
            for line in results:
                f.write(f"{line[0]} \t {line[1]} \t {line[2]} \t {line[3]} \t {line[4]}\n".encode("utf-8"))
        results = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Sensitivity")
    parser.add_argument("--dataset_name", type=str, default="snli",
                    help="Name of dataset to use")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_cache_dir", type=str, default="data",
                    help="Directory of dataset cache")
    parser.add_argument("--split", type=str, default="test",
                    help="Name of split to subsample from.")
    parser.add_argument("--n_samples", type=int, default=100,
                    help="Number of samples per class to use.")
    parser.add_argument("--n_neighbour_samples", type=int, default=10,
                    help="Number of subset replacements samples.")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nodes", type=int, default=1,
                    help="Number of nodes to use.")
    parser.add_argument("--gpus", type=int, default=1,
                    help="Number of gpus per node to use.")
    parser.add_argument("--node_rank", type=int, default=0,
                    help="Ranking of node to use.")
    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(x) for x in range(args["gpus"])])
    args["dist_url"] = f"tcp://localhost:8888"
    args["world_size"] = args["gpus"] * args["nodes"]
    mp.spawn(generate_neighbours, nprocs=args["gpus"], args=(args,))
