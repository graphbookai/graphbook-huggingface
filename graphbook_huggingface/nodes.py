import graphbook.steps as steps
from graphbook.resources import Resource
import torch
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
            "description": "The function called when model outputs are received from the pipeline. By default, you may use AssignModelOutputToNotes.",
        },
    }
    Outputs = ["out"]
    Category = "Huggingface"

    def __init__(
        self,
        id,
        model_id: str,
        batch_size: int,
        item_key: str,
        device_id: int,
        fp16: bool,
        log_model_outputs: bool,
        on_model_outputs: callable,
    ):
        super().__init__(id, batch_size, item_key)
        self.pipe = pipeline(
            model=model_id,
            batch_size=batch_size,
            device=device_id,
            torch_dtype=torch.float16 if fp16 else None,
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


default_convert_fn = \
"""def convert_fn(outputs, items, notes):
    for output, note in zip(outputs, notes):
        if note["model_output"] is None:
            note["model_output"] = []
        note["model_output"].append(output)
"""


class AssignModelOutputToNotes(Resource):
    """
    Used to assign the model outputs to its corresponding note. This is the default note creation function for Huggingface pipelines.
    """

    Parameters = {"val": {"type": "function", "default": default_convert_fn}}
    Category = "Huggingface"

    def __init__(self, val):
        super().__init__(val)
