import datasets
import os
import torch
from datasets import load_dataset
from typing import List


DATADIR = {
    "premise":{
        "matched":"text-adversarial-attack/adv_samples/bert_mnli_matched_premise_0-1000_iters=100_cw_kappa=5_lambda_sim=20.0_lambda_perp=1.pth",
        "mismatched":"text-adversarial-attack/adv_samples/bert_mnli_mismatched_premise_0-1000_iters=100_cw_kappa=5_lambda_sim=20.0_lambda_perp=1.pth"
    },

    "hypothesis":{
        "matched": "text-adversarial-attack/adv_samples/bert_mnli_matched_hypothesis_0-1000_iters=100_cw_kappa=5_lambda_sim=20.0_lambda_perp=1.pth",
        "mismatched": "text-adversarial-attack/adv_samples/bert_mnli_mismatched_hypothesis_0-1000_iters=100_cw_kappa=5_lambda_sim=20.0_lambda_perp=1.pth"
    }
}

class Gbda(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="premise",
            description="Whitebox attack with premise as target."),
        datasets.BuilderConfig(name="hypothesis",
            description="Whitebox attack with hypothesis as target.")
    ]


    def _info(self):
        features = datasets.Features(
            {
                "premise": datasets.Value("string"),
                "hypothesis": datasets.Value("string"),
                "label": datasets.Value("int32"),
                "adversarial_premise": datasets.Value("string"),
                "adversarial_hypothesis": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description = "",
            features = features,
            supervised_keys=None,
            homepage="https://github.com/facebookresearch/text-adversarial-attack",
            citation=""
        )


    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_dir = DATADIR[self.config.name]
        return [
            datasets.SplitGenerator(
                name="matched",
                gen_kwargs = {
                    "filepath": data_dir["matched"],
                    "split": "matched"
                }
            ),
            datasets.SplitGenerator(
                name="mismatched",
                gen_kwargs = {
                    "filepath": data_dir["mismatched"],
                    "split": "mismatched"
                }
            )
        ]


    def _generate_examples(self, filepath:str, split: str):
        ckpt = torch.load(filepath)
            
        for i in range(len(ckpt["labels"])):
            yield i, {
                "premise":ckpt["clean_texts"]["premise"][i],
                "hypothesis":ckpt["clean_texts"]["hypothesis"][i],
                "label":ckpt["labels"][i],
                "adversarial_premise":ckpt["adv_texts"]["premise"][i],
                "adversarial_hypothesis":ckpt["adv_texts"]["hypothesis"][i]
            }


if __name__ == "__main__":
    dataset = load_dataset("load_utils/load_gbda.py", name="premise")
    print(dataset["matched"][0])
