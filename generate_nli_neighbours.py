"""Code slightly adjusted from: https://github.com/m-hahn/sensitivity/blob/main/code/xlnet/GLUE/MNLI/generate18_c.py
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
from data_utils.load_configs import LOAD_CONFIGS
from data_utils.load_data import get_dataset
from torch.utils.data import Dataset, DataLoader
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
    model = AutoModelForCausalLM.from_pretrained("xlnet-large-cased")
    return model, tokenizer

def prepare_input_text(text:str):
    if text.endswith("."):
        text = text[:-1]
    
    text = text + " ."
    return text

def prepare_samples(args, tokenizer):
    """ Function which loads a class stratified subsample of chosen split
    of the dataset"""
    config = LOAD_CONFIGS[args["dataset_name"]]
    config["split"] = args["split"]
    dataset = get_dataset(config["dataset"], **config)
    dataset = dataset.shuffle(args["seed"])
    dataset = dataset.filter(lambda x: 0 <= x["label"] < 3)
    
    if "gold" in dataset.column_names:
        dataset = dataset.rename_column("gold", "label")

    dataset = dataset.map(lambda _, idx:{"idx":idx}, with_indices=True)
    dataset = concatenate_datasets(
        [
            dataset.filter(lambda x: x["label"] == class_).select(range(args["n_samples"]))
            for class_ in [0, 1, 2]
        ]
    )
    necessary_columns = ["premise", "hypothesis", "label", "idx"]
    # Remove any unnecessary columns
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in necessary_columns]
    )

    # make sure sequences end with .
    dataset = dataset.map(
        lambda x: {
            "premise": prepare_input_text(x["premise"]),
            "hypothesis": prepare_input_text(x["hypothesis"])
        }
    )
    # Tokenize samples
    dataset = dataset.map(
        lambda x: tokenizer(x["premise"] + "<sep>" + x["hypothesis"], add_special_tokens=False, return_tensors="pt")
    )
    return dataset

def generate_n_gram_subset_masks(n, length, token_strings, sep_idx):
    masks = set()
    for offset in range(length - n + 1):
        mask = ("0" * offset) + ("1" * n) + ("0" * (length - offset - n))
        masks.add(adjust_subset(mask, token_strings, sep_idx))
    return masks

def generate_span_subset_masks(sequence_length, token_strings, sep_idx):
    span_length = int(sequence_length/7) + 1
    subsets = []
    for subset_idx in range(1, 2**7-1):
        binary_mask = format(subset_idx, "b")
        binary_mask = ("0" * (7 - len(binary_mask))) + binary_mask
        binary_mask = "".join([x*span_length for x in binary_mask])[:sequence_length]
        subsets.append(adjust_subset(binary_mask, token_strings, sep_idx))
    return subsets

def adjust_subset_to_word_boundaries(subset, token_strings):
    subset = list(subset)
    last_start = 0
    # Last token is a punct
    subset[-1] = "0"
    for i in range(1, len(subset)-1):
        if token_strings[i].startswith("▁"):
            last_start = i
        if subset[i] == "1":
            # If subset starts at sub_word, extend it to start of current word
            if subset[i-1] == "0":
                if not token_strings[i].startswith("▁"):
                    for j in range(last_start, i):
                        subset[j] = "1"
            # If subset ends at sub_word, shorten it to end of previous word
            if i+2 < len(subset) and subset[i+1] == "0":
                if not token_strings[i+1].startswith("▁"):
                    for j in range(last_start, i+1):
                        subset[j] = "0"
    subset = "".join(subset)
    return subset

def adjust_subset_to_separation_idx(subset, separation_idx):
    subset = list(subset)
    subset[-1] = "0" #Not necessary,imo
    # NOTE: previously: subset[separation_idx-1] = "0"
    subset[separation_idx] = "0"
    if "1" not in subset:
        return
    return "".join(subset)

def adjust_subset(subset, token_string, separation_idx):
    subset = adjust_subset_to_word_boundaries(subset, token_string)
    return adjust_subset_to_separation_idx(subset, separation_idx)

def generate_example_subsets(example, tokenizer):
    input_ids = example["input_ids"][0]
    sep_idx = [i for i, x in enumerate(input_ids) if x == 4][0]
    #del input_ids[sep_idx]
    
    sequence_length = len(input_ids)
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    token_strings = [tokenizer.convert_ids_to_tokens(x) for x in input_ids]

    subsets = set()
    # In the paper they say they use up to 8-grams, but in their code they only use up to 6-grams.
    for n in range(1, 7):
        subsets.update(generate_n_gram_subset_masks(n, sequence_length, token_strings, sep_idx))

    subsets.update(generate_span_subset_masks(sequence_length, token_strings, sep_idx))
    return subsets

class ExampleSubsetsDataset:

    def __init__(self, examples, tokenizer, pad_text=PAD_TEXT):
        self.examples = examples
        self.tokenizer = tokenizer
        self.mask_idx = self.tokenizer.convert_tokens_to_ids("<mask>")
        self.pad_ids = self.tokenizer(pad_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.pad_length =self.pad_ids.size(0)
        self.data_ques = defaultdict(deque)

    def __len__(self):
        return len(self.examples)

    def encode_subset(self, original_input_ids, subset):
        subset = np.array([int(x) for x in subset])
        subset_ids = torch.clone(original_input_ids)
        subset_ids[self.pad_length:][subset == 1] = self.mask_idx
        return subset_ids

    def get_first_mask_idx(self, subsets_ids):
        mask_tokens = (subsets_ids == 6).int()
        idx = torch.arange(mask_tokens.size(1), 0, -1, device=subsets_ids.device)
        first_indices = torch.argmax(mask_tokens * idx, 1, keepdim=True)
        return first_indices

    def construct_permutation_mask(self, subsets_ids):
        permutation_mask = torch.zeros((subsets_ids.size(0), subsets_ids.size(1), subsets_ids.size(1)), dtype=torch.float, device=subsets_ids.device)
        permutation_mask[:, :, self.pad_length:][subsets_ids==6] = 1.0
        return permutation_mask

    def construct_target_mapping(self, subsets_ids, n_targets):
        mask_mask = torch.nonzero((subsets_ids == 6).long(), device=subsets_ids.device)
        target_idx = torch.arange(0, n_targets, device=subsets_ids.device).repeat(mask_mask.size(0)//n_targets)
        target_mapping = torch.zeros((subsets_ids.size(0), n_targets, subsets_ids.size(1)), dtype=torch.float, device=subsets_ids.device)
        target_mapping[mask_mask[:,0],target_idx, mask_mask[:,1]] = 1.0
        return target_mapping

    def construct_single_target_mapping(self, subsets_ids):
        first_target_idx = self.get_first_mask_idx(subsets_ids)
        target_mapping = torch.zeros((subsets_ids.size(0), subsets_ids.size(1)), dtype=torch.float, device=subsets_ids.device)
        target_mapping.scatter_(1, first_target_idx, 1.0)
        return target_mapping.unsqueeze(1)

    def store_item(self, idx):
        example = self.examples[idx]
        subsets = generate_example_subsets(example, tokenizer)

        original_input_ids = torch.LongTensor(example["input_ids"][0])
        original_input_ids = torch.cat((self.pad_ids.clone(), original_input_ids), dim=0)
        example_idx = torch.Tensor([example["idx"]]).long()
        for subset in subsets:
            if subset is not None:
                n_masks = subset.count("1")
                encoding = self.encode_subset(original_input_ids, subset)
                self.data_ques[n_masks].append((encoding, example_idx))


def to_device(batch, device):
    return [x.to(device) for x in batch]

def pad_batch(input_ids):
    max_len = max([len(x) for x in input_ids])
    padded_input_ids = torch.full((len(input_ids), max_len), 5, dtype=torch.long)
    attn_mask = torch.zeros(len(input_ids), max_len)
    for i in range(len(input_ids)):
        padded_input_ids[i, :len(input_ids[i])] = input_ids[i]
        attn_mask[i, :len(input_ids[i])] = 1
    return padded_input_ids, attn_mask

def batch_generator(dataset, batch_size):
    for n_targets in dataset.data_ques.keys():
        input_ids, example_ids = [], []
        while len(dataset.data_ques[n_targets]) > 0:
            sequence_ids, ex_id = dataset.data_ques[n_targets].pop()
            input_ids.append(sequence_ids)
            example_ids.append(ex_id)
            if len(input_ids) == batch_size or len(dataset.data_ques[n_targets]) == 0:
                input_ids, attn_mask = pad_batch(input_ids)
                yield (input_ids, attn_mask, example_ids)
                input_ids, example_ids = [], []

def generate_token(model, batch, vocab_mask, device):
    # Construct inputs
    input_ids, attn_mask, _ = batch
    permutation_mask = dataset.construct_permutation_mask(input_ids)
    target_mask = dataset.construct_single_target_mapping(input_ids)
    input_ids, attn_mask, permutation_mask, target_mask = to_device([input_ids, attn_mask, permutation_mask, target_mask], device)
    
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
    target_idx = dataset.get_first_mask_idx(input_ids)
    input_ids.scatter_(1, target_idx, next_tokens)

    return input_ids

def tokens_to_str(tokens, tokenizer, pad_length):
    strings = []
    for sequence in tokens:
        strings.append(tokenizer.decode(sequence[pad_length:], skip_special_tokens=False))
    return strings

@torch.no_grad()
def generate_neighbours(model, dataset, args):
    vocab_mask = torch.cuda.FloatTensor([float("-inf") if ("<" in dataset.tokenizer.convert_ids_to_tokens(x)) else 0 for x in range(32000)]).view(1, 1, -1)

    results = []
    model = model.to(args["device"])
    model.eval()
    data_generator = batch_generator(dataset, 64)
    for batch in data_generator:
        for _ in range(args["n_neighbour_samples"]):
            input_ids, attn_mask, example_ids = batch
            n_remaining_targets = torch.sum((input_ids[0] == 6).int())
            n_targets = n_remaining_targets.item()

            while n_remaining_targets > 0:
                input_ids = generate_token(model, (input_ids, attn_mask, example_ids), vocab_mask, args["device"])
                n_remaining_targets = torch.sum((input_ids[0] == 6).int())

            decoded = tokens_to_str(input_ids, dataset.tokenizer, dataset.pad_length)

            for i, neighbour in enumerate(decoded):
                premise, hypothesis = neighbour.replace("<pad>", "").split("<sep>")
                results.append((premise, hypothesis, n_targets, example_ids[i].item()))

        
    with open(args["save_dir"] + f"/{args['dataset_name']}/_n_samples={args['n_samples']}_split={args['split']}_seed={args['seed']}.txt", "w") as f:
        for line in results:
            f.write(f"{line[0]}, {line[1]}, {line[2]}, {line[3]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Sensitivity")
    parser.add_argument("--dataset_name", type=str, default="snli",
                    help="Name of dataset to use")
    parser.add_argument("--save_dir", type=str, default="results")
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
    parser.add_argument("--device", default=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    args = vars(parser.parse_args())
    model, tokenizer = load_model()
    samples = prepare_samples(args, tokenizer)
    dataset = ExampleSubsetsDataset(samples, tokenizer)
    
    for i in range(len(dataset)):
        dataset.store_item(i)
    generate_neighbours(model, dataset, args)
