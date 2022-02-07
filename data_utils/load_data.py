from datasets import load_dataset, load_from_disk
from .load_configs import LOAD_CONFIGS


def get_dataset(path: str, **kwargs: str):
    """Function which retrieves dataset specified by keyword arguments.
    If dataset is already saved at location specified by <path> it will load from said location.
    Else the dataset will be loaded via instructions in <**kwargs> argument.
    
    Args:
        path (str): location to load or save dataset to.
    
    Keyword Args:
        **name (str, optional): name of the configuration of the dataset.
        **load_script (str, optional): location of load_script.
        **split (str, optional): name of the split to be loaded.
    """
    if not "data_files" in kwargs.keys():
        return load_dataset(path=path, name=kwargs["name"], cache_dir=kwargs["cache_dir"], split=kwargs["split"])
    return load_dataset(path=path, name=kwargs["name"], cache_dir=kwargs["cache_dir"], split=kwargs["split"], data_files=kwargs["data_files"])

def load_all(load_configs):
    for dataset_name, config in load_configs.items():
        print(f"Loading {dataset_name}...")
        get_dataset(config["dataset"], **config)

    print("Finished loading all datasets!")        


if __name__ == "__main__":

    #test_args = {
    #    "name": "premise",
    #    "split": None,
    #    "load_script": None,
    #    "cache_dir": "data/gbda_mnli"
    #}

    #load_all(LOAD_CONFIGS)

    config = LOAD_CONFIGS["snli"]
    dataset = get_dataset(config["dataset"], **config)
    print(dataset["train"][:10])
    print(dataset["test"][:10])
