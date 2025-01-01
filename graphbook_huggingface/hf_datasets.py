from graphbook import Note
import graphbook.steps as steps
from datasets import load_dataset
import random


class HuggingfaceDataset(steps.GeneratorSourceStep):
    """
    Loads a dataset from 🤗 Hugging Face and yields each dataset row as a Note.
    The dataset is loaded using the `datasets.load_dataset` method.
    If loading an dataset with images, you must specify the columns that contain images if you want them to be displayed in the UI.
    
    Args:
        dataset_id (str): The dataset ID from Huggingface
        split (str): The split of the dataset to use
        shuffle (bool): Whether to shuffle the dataset
        log_data (bool): Whether to log the outputs as JSON to the node UI
        image_columns (List[str]): The columns in the dataset that contain images. This is to let Graphbook know how to display the images in the UI.
        kwargs (dict): Additional keyword arguments to pass to the internal Hugging Face datasets method `datasets.load_dataset` which gets used to load the dataset.
    """
    RequiresInput = False
    Parameters = {
        "dataset_id": {
            "type": "string",
            "description": "The dataset ID from Huggingface",
        },
        "split": {
            "type": "string",
            "default": "train",
            "description": "The split of the dataset to use",
        },
        "shuffle": {
            "type": "boolean",
            "default": False,
            "description": "Whether to shuffle the dataset",
        },
        "log_data": {
            "type": "boolean",
            "default": True,
            "description": "Whether to log the outputs as JSON to the node UI",
        },
        "image_columns": {
            "type": "list[string]",
            "description": "The columns in the dataset that contain images. This is to let Graphbook know how to display the images in the UI.",
            "required": False,
        },
        "kwargs": {
            "type": "dict",
            "description": "Additional keyword arguments to pass to the dataset",
            "required": False,
        },
    }
    Outputs = ["out"]
    Category = "Huggingface"

    def __init__(
        self,
        dataset_id: str,
        split: str,
        shuffle: bool,
        log_data: bool,
        image_columns=[],
        kwargs={},
    ):
        super().__init__()
        self.dataset = dataset_id
        self.split = split
        self.shuffle = shuffle
        self.log_data = log_data
        self.image_columns = image_columns
        self.kwargs = kwargs

    def get_note_dict(self, item):
        d = {}
        d.update(item)
        for col in self.image_columns:
            if col in item:
                d[col] = {"type": "image", "value": item[col]}
        return d
    
    def get_iterator(self, dataset):
        if self.shuffle:
            n = len(dataset)
            indices = random.sample(range(n), n)
            for i in indices:
                yield dataset[i]   
        else:
            for item in dataset:
                yield item

    def load(self):
        try:
            dataset = load_dataset(self.dataset, self.split, **self.kwargs)
        except ValueError:
            dataset = load_dataset(self.dataset, **self.kwargs)
        if isinstance(dataset, dict):
            dataset = dataset[self.split]
            
        dataset = self.get_iterator(dataset)
        for item in dataset:
            if self.log_data:
                self.log(item, "json")
            note = Note(self.get_note_dict(item))
            yield {"out": [note]}

