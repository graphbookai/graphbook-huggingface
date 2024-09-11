from graphbook import Note
import graphbook.steps as steps
from graphbook.resources import Resource
import torch
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.base import pad_collate_fn, no_collate_fn
from graphbook.utils import transform_function_string


class HuggingfacePipeline(steps.BatchStep):
    RequiresInput = True
    Parameters = {
        "model_id": {
            "type": "string",
            "description": "The model ID from Huggingface",
        },
        "batch_size": {
            "type": "number",
            "default": 8,
            "description": "The batch size for the pipeline",
        },
        "item_key": {
            "type": "string",
            "description": "The key in the input item to use as input. Should be reference a URL linking to an image, a base64 string, a local path, or a PIL image.",
        },
        "device_id": {
            "type": "number",
            "default": 0,
            "description": "The GPU ID to use for the pipeline",
        },
        "fp16": {
            "type": "boolean",
            "default": False,
            "description": "Whether to use fp16",
        },
        "log_model_outputs": {
            "type": "boolean",
            "default": True,
            "description": "Whether to log the model outputs as JSON to the node UI",
        },
        "on_model_outputs": {
            "type": "resource",
            "description": "The function called when model outputs are received from the pipeline. By default, you may use CreateNotesFromModelOutputs.",
            "required": False,
        },
        "kwargs": {
            "type": "dict",
            "description": "Additional keyword arguments to pass to the model pipeline",
            "required": False,
        },
    }
    Outputs = ["out"]
    Category = "Huggingface"

    def __init__(
        self,
        model_id: str,
        batch_size: int,
        item_key: str,
        device_id: int,
        fp16: bool,
        log_model_outputs: bool,
        on_model_outputs: callable,
        kwargs: dict = {},
    ):
        super().__init__(batch_size, item_key)
        self.pipe = pipeline(
            model=model_id,
            batch_size=batch_size,
            device=device_id,
            torch_dtype=torch.float16 if fp16 else None,
            **kwargs
        )
        self.log_model_outputs = log_model_outputs
        self.create_note_fn = transform_function_string(on_model_outputs)
        preprocess_params, forward_params, postprocess_params = (
            self.pipe._sanitize_parameters()
        )

        self.preprocess_params = {**self.pipe._preprocess_params, **preprocess_params}
        self.forward_params = {**self.pipe._forward_params, **forward_params}
        self.postprocess_params = {
            **self.pipe._postprocess_params,
            **postprocess_params,
        }

        self.feature_extractor = (
            self.pipe.feature_extractor
            if self.pipe.feature_extractor is not None
            else self.pipe.image_processor
        )

    def load_fn(self, item: dict):
        im = self.pipe.preprocess(item["value"], **self.preprocess_params)
        return im

    @torch.no_grad()
    def on_item_batch(self, inputs, items, notes):
        # Will be moved to worker pool soon
        collate_fn = (
            no_collate_fn
            if len(inputs) <= 1
            else pad_collate_fn(self.pipe.tokenizer, self.feature_extractor)
        )
        batch = collate_fn(inputs)
        outputs = self.pipe.forward(batch, **self.forward_params)
        outputs = self.pipe.postprocess(outputs, **self.postprocess_params)
        if self.log_model_outputs:
            self.log(outputs, "json")
        if self.create_note_fn:
            self.create_note_fn(outputs, items, notes)


class HuggingfaceDataset(steps.GeneratorSourceStep):
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
        log_data: bool,
        image_columns=[],
        kwargs={},
    ):
        super().__init__()
        self.dataset = dataset_id
        self.split = split
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

    def load(self):
        try:
            dataset = load_dataset(self.dataset, self.split, **self.kwargs)
        except ValueError:
            dataset = load_dataset(self.dataset, **self.kwargs)
        if isinstance(dataset, dict):
            dataset = dataset[self.split]
        for item in dataset:
            if self.log_data:
                self.log(item, "json")
            note = Note(self.get_note_dict(item))
            yield {"out": [note]}


default_convert_fn = """def convert_fn(outputs, items, notes):
  for output, note in zip(outputs, notes):
    if note["model_output"] is None:
      note["model_output"] = []
    note["model_output"].append(output)"""


class AssignModelOutputsToNotes(Resource):
    """
    Used to assign the model outputs to its corresponding note. This is the default note creation function for Huggingface pipelines.
    """

    Parameters = {"val": {"type": "function", "default": default_convert_fn}}
    Category = "Huggingface"

    def __init__(self, val):
        super().__init__(val)
