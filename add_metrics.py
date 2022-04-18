import numpy as np
import spacy, benepar
import torch
from analyse_utils import NliAnalyzer, whitespace_tokenizer, map_labels
from analyse_utils import get_example_overlap, remove_punct
from train_snli import prepare_model

from collections import deque
from sentence_transformers import SentenceTransformer, util
from datasets import concatenate_datasets, DatasetDict
from datasets.utils.logging import set_verbosity_error

def add_cartography_data(analyzer, dataset, model, values, seeds):
    paths = []
    prefixes = []
    for seed in seeds:
        paths.append(f"results/{dataset}/{model}_{values['split_groups']['train']}_epoch=5_seed={seed}_batchsize=32_betas=(0.9, 0.999)lr=1e-05_weight_decay=0.01_warmup_steps=0.06")
        prefixes.append(f"{model}_{values['split_groups']['train']}_seed={seed}")
    
    analyzer.processor.add_cartography_data(values["split_groups"], paths, prefixes)


def add_pvi_data(analyzer, dataset, model, values, seeds):
    paths = []
    prefixes = []
    for seed in seeds:
        paths.append(f"results/{dataset}/{model}_{values['split_groups']['train']}_epoch=5_seed={seed}_batchsize=32_betas=(0.9, 0.999)lr=1e-05_weight_decay=0.01_warmup_steps=0.06")
        prefixes.append(f"{model}_{values['split_groups']['train']}_seed={seed}")
    analyzer.processor.add_pvi_data(values["split_groups"], paths, prefixes)

def add_tree_based_metrics(analyzer, values, model):

    def parse_examples(examples, columns, **kwargs):
        results = {}
        for col in columns:
            docs = list(model.pipe(examples[col]))
            yngve = get_example_yngve(docs)
            parse_height = get_parse_tree_height(docs)
            hans = get_hans_heuristics(docs, examples, columns, **kwargs)

            results[col + "_yngve"] = yngve
            results[col + "_parse_height"] = parse_height
            results[col + "_lexical_heuristic"] = hans["lexical_heuristic"]
            results[col + "_subsequence_heuristic"] = hans["subsequence_heuristic"]
            results[col + "_constituence_heuristic"] = hans["constituence_heuristic"]
             
        return results

    def get_example_yngve(docs):

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

        return [avg_yngve_score(doc) for doc in docs]

    def get_parse_tree_height(docs):

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

        return [get_height(doc) for doc in docs]

    def get_hans_heuristics(docs, examples, columns, **kwargs):

        def find_lexical_heuristic(examples, columns):
            return [
                1 if set(col_1).issubset(col_0) else 0
                for col_1, col_0 in zip(examples[columns[1]], examples[columns[0]])
            ]

        def find_subsequence_heuristic(examples, columns):
            return [
                1 if " ".join(col_1) in " ".join(col_0) else 0
                for col_1, col_0 in zip(examples[columns[1]], examples[columns[0]])
            ]

        def find_constituence_heuristic(docs, examples, idx, columns):
            results = [0 for _ in range(len(examples[columns[0]]))]

            for i in idx:
                doc = docs[i]
                text = examples[columns[1]][i]
                #consts = []
                #for sent in doc.sents:
                #    consts.extend([c.text.lower for c in sent._.constituents])
                constituents = [c.text.lower() for sent in doc.sents for c in sent._.constituents]
                results[i] = 1 if text.lower() in constituents else 0
            return results

        results_lexical = find_lexical_heuristic(examples, columns)
        results_subsequence = find_subsequence_heuristic(examples, columns)
        subsequence_ids = [i for i in range(len(results_subsequence)) if results_subsequence[i] == 1]
        results_const = find_constituence_heuristic(
            docs,
            examples, 
            subsequence_ids, 
            [columns[1], columns[0]]
        )
    
        return {
            "lexical_heuristic": results_lexical,
            "subsequence_heuristic": results_subsequence,
            "constituence_heuristic": results_const
        }

        
    # add sent objects to dataset
    analyzer.processor.apply_operation(values["columns"], parse_examples, batched=True, model=model)

def add_premise_hypothesis_similarity(analyzer, values, model):

    def add_embeddings(examples, columns, model, **kwargs):
        return {
            col + "_sent_embedding": model.encode(examples[col])
            for col in columns
        }

    def add_cosine_sim(examples, columns, **kwargs):
        cos_sim = torch.diagonal(
            util.cos_sim(
                examples[columns[0] + "_sent_embedding"],
                examples[columns[1] + "_sent_embedding"]
            ), 0
        ).tolist()
        return {
            "premise_hypothesis_cosine_sim": cos_sim
        }

    def add_token_overlaps(examples, columns, **kwargs):
        return {
            "tokens_whitespace_premise_hypothesis_overlap": [
                len(set(premise) & set(hypothesis)) / len(set(hypothesis))
                for premise, hypothesis in zip(examples[columns[0]], examples[columns[1]])
            ]
        }

    analyzer.processor.apply_operation(
        values["columns"],
        add_embeddings,
        batched=True,
        model=model
    )

    analyzer.processor.apply_operation(
        values["columns"],
        add_cosine_sim,
        batched=True
    )

    analyzer.processor.apply_operation(
        values["columns"],
        add_token_overlaps,
        batched=True
    )

def add_ppl(analyzer, values, model, tokenizer):

    def get_ppl_example(examples, columns, **kwargs):
        losses = get_metrics(
            kwargs["model"],
            kwargs["tokenizer"],
            examples[columns[0]],
            examples[columns[1]],
            torch.device("cuda")
        )
        return {
            "premise ppl": losses[0],
            "hypothesis_ppl": losses[1]
        }

    analyzer.processor.apply_operation(
        values["columns"],
        get_ppl_example,
        batched=True,
        batch_size=16,
        model=model,
        tokenizer=tokenizer
    )
    
def load_metric_model(metric):
    if metric == "parse_tree":
        model = spacy.load('en_core_web_md')
        model.add_pipe("benepar", config={"model": "benepar_en3"})
        return model
    elif metric == "cosine":
        model = SentenceTransformer("all-mpnet-base-v2")
        return model
    elif metric == "perplexity":
        model, tokenizer = prepare_model("gpt2_large")
        tokenizer.pad_token = "<|PAD|>"
        model = model.to("cuda")
        model.eval()
        return model, tokenizer

def load_analyzer(config):
    dataset = list(config.keys())[0]
    values = config[dataset]
    save_dir = f"results/{dataset}"
    analyzer = NliAnalyzer(
        dataset,
        save_dir,
        whitespace_tokenizer,
        "whitespace",
        subsample_size = 0,
        correctness = "both"
    )

    if dataset == "wanli":
        analyzer.processor.apply_operation("gold", map_labels, batched=True)

    analyzer.processor.dataset = analyzer.processor.dataset.filter(
        lambda example: example["label"] >= 0
    )
    
    # Cartography & PVI
    add_cartography_data(analyzer, dataset, "roberta", values, [42])
    add_pvi_data(analyzer, dataset, "roberta", values, [42])

    # Sequence_lengths
    analyzer.get_sequence_lengths(values["columns"], batched=True)

    # Yngve, Tree height, Hans Heuristics
    model = load_metric_model("parse_tree")
    add_tree_based_metrics(analyzer, values, model)
    del model

    # Premise - Hypothesis token overlap & Cosine Similarity
    model = load_metric_model("cosine")
    add_premise_hypothesis_similarity(analyzer, values, model)
    del model

    # Perplexity
    model, tokenizer = load_metric_model("perplexity")
    add_ppl(analyzer, values, model, tokenizer)
    del model
    del tokenizer

    analyzer.processor.dataset.save_to_disk(f"data/metrics/{dataset}")

def build_batch_inputs(tokenizer, sequences, device):
    inputs = tokenizer(
        ["<|endoftext|> " + s for s in sequences],
        padding="longest",
        return_tensors = "pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

@torch.no_grad()
def get_metrics(model, tokenizer, premises, hypotheses, device):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
    losses = [[], []]
    for i, sequences in enumerate([premises, hypotheses]):
        inputs = build_batch_inputs(tokenizer, sequences, device)
        outputs = model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(len(premises), -1)
        
        for j in range(len(sequences)):
            cross_entropy = torch.mean(loss[j][shift_labels[j] != tokenizer.pad_token_id])
            losses[i].append(torch.exp(cross_entropy).item())

    return losses     

if __name__ == "__main__":
    set_verbosity_error()

    configs = [
        {
            "cad_snli_combined": {
                "columns": ["premise", "hypothesis"],
                "splits": ["train", "validation", "test"],
                "split_groups": {
                    "train": ["train"],
                    "validation": ["validation"],
                    "test": ["test"]
                }
            }
        }
    ]

    load_analyzer(configs[0])
