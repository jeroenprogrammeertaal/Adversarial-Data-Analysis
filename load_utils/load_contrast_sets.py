import datasets
import os
import json
from datasets import load_dataset
from typing import List

CONTRAST_URLS = {
    "ropes": {
        "test": "https://raw.githubusercontent.com/allenai/contrast-sets/main/ropes/data/ropes_contrast_set_032820.json"
    },
    "quoref": {
        "test": "https://raw.githubusercontent.com/allenai/contrast-sets/main/quoref/quoref_test_perturbations_20191206_merged.json"
    },
    "drop": {
        "test": "https://raw.githubusercontent.com/allenai/contrast-sets/main/DROP/drop_contrast_sets_test.json"
    },
    "boolq": {
        "test": "https://raw.githubusercontent.com/allenai/contrast-sets/main/BoolQ/boolq_perturbed.json"
    }
}

class ContrastSets(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="ropes",  
                description="Perturbed instances of ropes test set."),
        datasets.BuilderConfig(name="quoref",
                description="Perturbed instances of quoref test set."),
        datasets.BuilderConfig(name="drop",
                description="Perturbed instances of drop test set."),
        datasets.BuilderConfig(name="boolq",
                description="Perturbed instances of boolq test set.")
    ]

    DEFAULT_CONFIG_NAME = "ropes"    

    def _info(self):
        if self.config.name == "boolq":
            features = datasets.Features(
                {
                    "paragraph": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "original_question": datasets.Value("string"),
                    "answer": datasets.Value("bool"),
                    "original_answer": datasets.Value("bool")
                }
            )
        elif self.config.name == "drop":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "passage": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "original_question": datasets.Value("string"),
                    "answer": datasets.Sequence(
                        {
                            "number": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "day": datasets.Value("string"),
                            "month": datasets.Value("string"),
                            "year": datasets.Value("string")
                        }
                    )
                }
            )

        elif self.config.name == "ropes":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "background": datasets.Value("string"),
                    "situation": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Value("string")
                }
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32")
                        }
                    )
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
        urls = CONTRAST_URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test"
                }
            )
        ]

    def _generate_boolq_examples(self, filepath: str, split: str):
        key = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            # First element is not a datapoint
            for article in data["data"][1:]:
                title = article["title"]
                paragraph = article["paragraph"]
                original_question = article["question"]
                original_answer = article["answer"]
                for qa in article["perturbed_questions"]:
                    yield key, {
                        "paragraph": paragraph,
                        "original_question": original_question,
                        "original_answer": original_answer,
                        "question": qa["perturbed_q"],
                        "answer": qa["answer"]
                    }
                    key += 1


    def _generate_drop_examples(self, filepath: str, split: str):
        key = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for title in data:
                article = data[title]
                passage = article["passage"]
                for qa in article["qa_pairs"]:
                    answer = qa["answer"]
                    yield key, {
                        "id": qa["query_id"],
                        "passage": passage,
                        "question": qa["question"],
                        "original_question": qa["original_question"],
                        "answer": {
                            "number": answer["number"],
                            "text": answer["spans"],
                            "day": answer["date"]["day"],
                            "month": answer["date"]["month"],
                            "year": answer["date"]["year"]
                        }
                    }
                    key += 1

    
    def _generate_ropes_examples(self, filepath: str, split: str):
        key = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for article in data["data"]:
                for paragraph in article["paragraphs"]:
                    background = paragraph["background"]
                    situation = paragraph["situation"]
                    for qa in paragraph["qas"]:
                        answers = [answer["text"] for answer in qa["answers"]]

                        yield key, {
                            "id": qa["id"],
                            "background": background,
                            "situation": situation,
                            "question": qa["question"],
                            "answers": answers
                        }
                        key += 1

    def _generate_quoref_examples(self, filepath:str, split:str):
        key = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for article in data["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"] for answer in qa["answers"]]
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.

                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1


    def _generate_examples(self, filepath:str, split:str):
        if self.config.name == "quoref":
            return self._generate_quoref_examples(filepath, split)
        elif self.config.name == "ropes":
            return self._generate_ropes_examples(filepath, split)
        elif self.config.name == "drop":
            return self._generate_drop_examples(filepath, split)
        else:
            return self._generate_boolq_examples(filepath, split)


if __name__ == "__main__":
    dataset = load_dataset("load_utils.py", name="all_combined")
