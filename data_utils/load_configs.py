
LOAD_CONFIGS = {
    "snli": {
        "dataset": "snli",
        "name": None,
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/snli"
    },
    "mnli": {
        "dataset": "multi_nli",
        "name": None,
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/mnli"
    },
    "anli": {
        "dataset": "anli",
        "name": None,
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/anli"
    },
    "wanli": {
        "dataset": "json",
        "name": None,
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/wanli",
        "data_files": {"train": "/media/jeroen/Extreme SSD/data/wanli/wanli/train.jsonl",
                        "test": "/media/jeroen/Extreme SSD/data/wanli/wanli/test.jsonl"}
    },
    "gbda_premise": {
        "dataset": "data_utils/load_gbda.py",
        "name": "premise",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/gbda"
    },
    "gbda_hypothesis": {
        "dataset": "data_utils/load_gbda.py",
        "name": "hypothesis",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/gbda"
    },
    "cad_snli_combined": {
        "dataset": "data_utils/load_cad_snli.py",
        "name": "all_combined",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/cad_snli"
    },
    "cad_snli_original": {
        "dataset": "data_utils/load_cad_snli.py",
        "name": "original",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/cad_snli"
    },
    "cad_snli_revised_combined": {
        "dataset": "data_utils/load_cad_snli.py",
        "name": "revised_combined",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/cad_snli"
    },
    "cad_snli_revised_hypothesis": {
        "dataset": "data_utils/load_cad_snli.py",
        "name": "revised_hypothesis",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/cad_snli"
    },
    "cad_snli_revised_premise": {
        "dataset": "data_utils/load_cad_snli.py",
        "name": "revised_premise",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/cad_snli"
    },
    "mrqa": {
        "dataset": "mrqa",
        "name": None,
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/mrqa"
    },
    "boolq": {
        "dataset": "boolq",
        "name": None,
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/boolq"
    },
    "aqa_bert_fooled": {
        "dataset": "load_utils/load_aqa_bert_electra.py",
        "name": "bert_fooled",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/aqa_be" 
    },
    "aqa_bert_nomodel": {
        "dataset": "load_utils/load_aqa_bert_electra.py",
        "name": "bert_nomodel",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/aqa_be"
    },
    "aqa_bert_random": {
        "dataset": "load_utils/load_aqa_bert_electra.py",
        "name": "bert_random",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/aqa_be"
    },
    "aqa_electra_fooled": {
        "dataset": "load_utils/load_aqa_bert_electra.py",
        "name": "electra_fooled",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/aqa_be"
    },
    "aqa_electra_nomodel": {
        "dataset": "load_utils/load_aqa_bert_electra.py",
        "name": "electra_nomodel",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/aqa_be"
    },
    "aqa_electra_random": {
        "dataset": "load_utils/load_aqa_bert_electra.py",
        "name": "electra_random",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/aqa_be"
    },
    "squad_adversarial_add_sent": {
        "dataset": "squad_adversarial",
        "name": "AddSent",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/squad_adversarial"
    },
    "squad_adversarial_add_one_sent": {
        "dataset": "squad_adversarial",
        "name": "AddOneSent",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/squad_adversarial"
    },
    "adversarial_qa_aqa": {
        "dataset": "adversarial_qa",
        "name": "adversarialQA",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/adversarial_qa"
    },
    "adversarial_qa_bidaf": {
        "dataset": "adversarial_qa",
        "name": "dbidaf",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/adversarial_qa"
    },
    "adversarial_qa_bert": {
        "dataset": "adversarial_qa",
        "name": "dbert",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/adversarial_qa"
    },
    "adversarial_qa_roberta": {
        "dataset": "adversarial_qa",
        "name": "droberta",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/adversarial_qa"
    },
    "contrast_sets_ropes": {
        "dataset": "load_utils/load_contrast_sets.py",
        "name": "ropes",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/contrast_sets" 
    },
    "contrast_sets_quoref": {
        "dataset": "load_utils/load_contrast_sets.py",
        "name": "quoref",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/contrast_sets"
    },
    "contrast_sets_drop": {
        "dataset": "load_utils/load_contrast_sets.py",
        "name": "drop",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/contrast_sets"
    },
    "contrast_sets_boolq": {
        "dataset": "load_utils/load_contrast_sets.py",
        "name": "boolq",
        "split": None,
        "cache_dir": "/media/jeroen/Extreme SSD/data/contrast_sets"
    }
}
