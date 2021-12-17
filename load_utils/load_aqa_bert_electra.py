import datasets
import os
import json
from datasets import load_dataset
from typing import List

AQA_BE_URLS = {
    "bert_fooled": {
        "train": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/fooled/train_squad.json",
        "validation": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/fooled/val_squad.json",
        "test": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/fooled/test_squad.json"
    },
    "bert_nomodel": {
        "train": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/nomodel/train_squad.json",
        "validation": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/nomodel/val_squad.json",
        "test": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/nomodel/test_squad.json"
    },
    "bert_random": {
        "train": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/random/train_squad.json",
        "validation": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/random/val_squad.json",
        "test": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/bert/random/test_squad.json"
    },
    "electra_fooled": {
        "train": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/fooled/train_squad.json",
        "validation": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/fooled/val_squad.json",
        "test": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/fooled/test_squad.json"
    },
    "electra_nomodel": {
        "train": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/nomodel/train_squad.json",
        "validation": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/nomodel/val_squad.json",
        "test": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/nomodel/test_squad.json"
    },
    "electra_random":{
        "train": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/random/train_squad.json",
        "validation": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/random/val_squad.json",
        "test": "https://raw.githubusercontent.com/facebookresearch/aqa-study/main/electra/random/test_squad.json"
    }
}

class AqaBertElectra(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="bert_fooled",  
                description="Contexts and questions that fooled BERT."),
        datasets.BuilderConfig(name="bert_nomodel",
                description="Random questions and contexts from SDC."),
        datasets.BuilderConfig(name="bert_random",
                description="Random questions and contexts."),
        datasets.BuilderConfig(name="electra_fooled",
                description="Contexts and questions that fooled Electra."),
        datasets.BuilderConfig(name="electra_random",
                description="Random questions and contexts"),
        datasets.BuilderConfig(name="electra_nomodel",
                description="Random questions and contexts from SDC.")
    ]

    DEFAULT_CONFIG_NAME = "bert_fooled"    

    def _info(self):
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
            homepage="https://github.com/facebookresearch/aqa-study",
            citation=""
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = AQA_BE_URLS[self.config.name]
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
                        if None in answer_starts:
                            continue

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


if __name__ == "__main__":
    dataset = load_dataset("load_utils.py", name="all_combined")
