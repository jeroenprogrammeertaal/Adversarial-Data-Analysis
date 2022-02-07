import csv
import pickle
import spacy, benepar
from analyse_utils import NliAnalyzer, whitespace_tokenizer, map_labels
from sentence_transformers import SentenceTransformer, util


def analyze_sequence_lengths(analyzer, columns, splits):
    analyzer.get_sequence_lengths(columns, batched=True)
    analyzer.plot_sequence_lengths(columns, splits)

def analyze_column_overlap(analyzer, column_1, column_2, splits):
    analyzer.get_overlaps(column_1, column_2, batched=True)
    analyzer.plot_attribute_density_per_class(
        splits,
        column_1,
        column_2,
        1,
        "overlap",
        f"_tokens_{analyzer.tokenizer_name}_{column_1}_{column_2}_overlap"
    )

def analyze_unique_top_n_grams(analyzer, columns, n, splits):
    counts = []
    for i in range(1, n+1):
        analyzer.get_n_grams(columns, i, batched=True)
        counts.append(analyzer.get_unique_n_grams(splits, columns, i))
        
    affix = f"/{splits}_{analyzer.tokenizer_name}_unique_grams" + analyzer.get_save_affix() + ".csv"
    with open(analyzer.save_dir + affix, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["gram"] + columns)
        for i, grams in enumerate(counts):
            writer.writerow([i] + [grams[column] for column in columns])


def analyze_cosine_sim(analyzer, column_1, column_2, splits):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    analyzer.get_cos_sim(column_1, column_2, model, batched=True)
    analyzer.plot_attribute_density_per_class(
        splits,
        column_1,
        column_2,
        1,
        "cosine_sim",
        f"{column_1}_{column_2}_cosine_sim_small_lm"
    )

def analyze_z_statistics(analyzer, columns, splits):
    analyzer.plot_artefacts(columns, splits)

def analyze_interexample_similarity(analyzer, group_column, sim_column, splits):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    analyzer.plot_interexample_similarity(
        group_column,
        sim_column,
        splits,
        model
    )

def analyze_hans_heuristics(analyzer, column_1, column_2, splits):
    model = spacy.load("en_core_web_md")
    model.add_pipe("benepar", config={"model": "benepar_en3"})
    analyzer.get_hans_heuristics(column_1, column_2, model)
    analyzer.plot_hans_heuristics(splits)

def test_constituency_parse():
    benepar.download('benepar_en3')

    model = spacy.load('en_core_web_md')
    model.add_pipe("benepar", config={"model": "benepar_en3"})

    test_sents = ["If the artist slept, the actor ran.",
                "Another test sentence, with many constituents"]

    docs = model.pipe(test_sents)
    for doc in docs:
        sent = list(doc.sents)[0]
        for x in sent._.constituents:
            print(x)

def subsample_mnli_pair_ids(analyzer):
    with open("pair_ids.pkl", "rb") as f:
        pair_ids = [int(x) for x in pickle.load(f)]
    
    return analyzer.processor.dataset.filter(lambda x: int(x["promptID"]) in pair_ids)

def analyse(config:dict):
    for dataset, values in config.items():
        save_dir = f"results/{dataset}"
        analyzer = NliAnalyzer(
            dataset,
            save_dir,
            whitespace_tokenizer,
            "whitespace",
            subsample_size=values["subsample_size"],
            correctness=values["correctness"]
        )
        if dataset == "wanli":
            analyzer.processor.apply_operation("gold", map_labels, batched=True)
        
        splits = values["splits"]
        
        analyzer.processor.dataset = subsample_mnli_pair_ids(analyzer)
        analyzer.save_dir = "results/mnli_wanli"
        #print(analyzer.processor.dataset)
        analyze_sequence_lengths(analyzer, values["columns"], splits)
        analyze_column_overlap(analyzer, values["columns"][0], values["columns"][1], splits)
        analyze_unique_top_n_grams(analyzer, values["columns"], 2, splits)
        analyze_cosine_sim(analyzer, values["columns"][0], values["columns"][1], splits)
        analyze_z_statistics(analyzer, values["columns"], splits)
        #analyze_interexample_similarity(analyzer, values["columns"][0], values["columns"][1], splits)
        analyze_hans_heuristics(analyzer, values["columns"][0], values["columns"][1], splits)

if __name__ == "__main__":
    analyse_configs = {
        #"snli": {
        #    "columns": ["premise", "hypothesis"],
        #    "subsample_size": 0,
        #    "splits": ["train"],
        #    "correctness": "both"
        #},
        "mnli": {
            "columns": ["premise", "hypothesis"],
            "subsample_size": 0,
            "splits": ["train"],
            "correctness": "both"
        },
        #"anli": {
        #    "columns": ["premise", "hypothesis"],
        #    "subsample_size": 0,
        #    "splits": ["train_r1", "train_r2", "train_r3"],
        #    "correctness": "incorrect"
        #},
        #"wanli": {
        #    "columns": ["premise", "hypothesis"],
        #    "subsample_size": 0,
        #    "splits": ["train"],
        #    "correctness": "both"
        #}
        #"gbda_premise": {
        #    "columns": ["adversarial_premise", "adversarial_hypothesis"],
        #    "subsample_size": 0,
        #    "correctness": "both"
        #},
        #"gbda_hypothesis": {
        #    "columns": ["adversarial_premise", "adversarial_hypothesis"],
        #    "subsample_size": 0,
        #    "correctness": "both"
        #},
        #"cad_snli_combined": {
        #    "columns": ["premise", "hypothesis"],
        #    "subsample_size": 0,
        #    "splits": ["train"],
        #    "correctness": "both"
        #},
        #"cad_snli_original": {
        #    "columns": ["premise", "hypothesis"],
        #    "subsample_size": 0,
        #    "correctness": "both"
        #},
        #"cad_snli_revised_combined": {
        #    "columns": ["premise", "hypothesis"],
        #    "subsample_size": 0,
        #    "correctness": "both"
        #},
        #"cad_snli_revised_hypothesis": {
        #    "columns": ["premise", "hypothesis"],
        #    "subsample_size": 0,
        #    "correctness": "both"
        #},
        #"cad_snli_revised_premise": {
        #    "columns":["premise", "hypothesis"],
        #    "subsample_size": 0
        #    "correctness": "both"
        #}
    }
    analyse(analyse_configs)
