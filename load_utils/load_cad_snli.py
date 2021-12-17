import datasets
import os
from datasets import load_dataset
from typing import List

CADSNLI_URLS = {
    "original": {
        "train": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/original/train.tsv",
        "validation": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/original/dev.tsv",
        "test": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/original/test.tsv"
    },
    "all_combined": {
        "train": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/all_combined/train.tsv",
        "validation": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/all_combined/dev.tsv",
        "test": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/all_combined/test.tsv"
    },
    "revised_combined": {
        "train": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_combined/train.tsv",
        "validation": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_combined/dev.tsv",
        "test": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_combined/test.tsv"
    },
    "revised_hypothesis":{
        "train": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_hypothesis/train.tsv",
        "validation": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_hypothesis/dev.tsv",
        "test": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_hypothesis/test.tsv"
    },
    "revised_premise": {
        "train": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_premise/train.tsv",
        "validation": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_premise/dev.tsv",
        "test": "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_premise/test.tsv"
    }
}

class CadSnli(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="all_combined",  
                description="Original and all revised parts combined."),
        datasets.BuilderConfig(name="original",
                description="Original examples from SNLI"),
        datasets.BuilderConfig(name="revised_combined",
                description="Revised Hypothesis and Premise combined."),
        datasets.BuilderConfig(name="revised_hypothesis",
                description="Revised hypothesis examples."),
        datasets.BuilderConfig(name="revised_premise",
                description="Revised Premise examples.")
    ]

    DEFAULT_CONFIG_NAME = "all_combined"    

    def _info(self):
        features = datasets.Features(
            {
                "premise": datasets.Value("string"),
                "hypothesis": datasets.Value("string"),
                "label": datasets.Value("int32")
            }
        )
        return datasets.DatasetInfo(
            description = "",
            features = features,
            supervised_keys=None,
            homepage="https://github.com/acmi-lab/counterfactually-augmented-data",
            citation=""
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = CADSNLI_URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["validation"],
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, filepath:str, split:str):
        label_to_int = {
            "entailment":0,
            "neutral":1,
            "contradiction":2
        }
        with open(filepath, encoding="utf-8") as f:
            # Skip header row
            next(f)
            for id_, row in enumerate(f):
                data = row.replace("\n", "").split("\t")
                yield id_, {
                    "premise": data[0],
                    "hypothesis": data[1],
                    "label": label_to_int[data[2]]
                }


if __name__ == "__main__":
    dataset = load_dataset("load_utils.py", name="all_combined")
