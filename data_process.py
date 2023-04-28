from pathlib import Path
from datasets import Dataset, concatenate_datasets, DatasetDict
import json


def load_dataset():
    def pair_data_iter(raw_data_dir: str):
        """
        Inputs
        ------
            raw_data_dir: str
                raw_data_dir

        Returns
        ------
            a json data {"prompt": str, "completion": str}
        """
        raw_data_dir = Path(raw_data_dir)
        for file_path in raw_data_dir.glob("*.json*"):
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    yield {"content": data["prompt"] + data["completion"]}


    def blog_data_iter(raw_data_dir: str):
        """
        Inputs
        ------
            raw_data_dir: str
                raw_data_dir

        Returns
        ------
            a json data {"prompt": str, "completion": str}
        """
        raw_data_dir = Path(raw_data_dir)
        for file_path in raw_data_dir.glob("*data.jsonl"):
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    yield {"content": data["title"] + data["text"]}



    def forum_data_iter(raw_data_dir: str):
        """
        Inputs
        ------
            raw_data_dir: str
                raw_data_dir

        Returns
        ------
            a json data {"prompt": str, "completion": str}
        """
        raw_data_dir = Path(raw_data_dir)
        for file_path in raw_data_dir.glob("*data.jsonl"):
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    content = ""
                    if len(data["qas"]) < 2:
                        continue
                    for qa in data["qas"]:
                        content += qa["text"]
                    yield {"content": content}


    # Load Training data
    train_manual_dataset = Dataset.from_generator(pair_data_iter, gen_kwargs={"raw_data_dir": "./training_data/manual"})
    train_xscript_dataset = Dataset.from_generator(pair_data_iter, gen_kwargs={"raw_data_dir": "./training_data/xscript/"})
    train_blog_dataset = Dataset.from_generator(blog_data_iter, gen_kwargs={"raw_data_dir": "./training_data/blog/"})
    train_forum_dataset = Dataset.from_generator(forum_data_iter, gen_kwargs={"raw_data_dir": "./training_data/forum/"})
    train_dataset = concatenate_datasets([train_manual_dataset, 
                                          train_xscript_dataset, 
                                          train_blog_dataset, 
                                          train_forum_dataset])

    # Load Validation data
    # no validation data


    # Combine train and valid dataset
    raw_datasets = DatasetDict(
        {
            "train": train_dataset,
        }
    )
    return raw_datasets