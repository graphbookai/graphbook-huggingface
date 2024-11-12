from typing import List
import graphbook.steps as steps
import torch
from transformers import pipeline
from transformers.pipelines.base import pad_collate_fn, no_collate_fn
from transformers.utils.generic import ModelOutput
from transformers.pipelines.pt_utils import PipelineIterator
import os


class NLPPipeline(steps.BatchStep):
    RequiresInput = True
    Parameters = {
        "task": {
            "type": "string",
            "description": "A task name from the Huggingface pipeline library",
            "required": False,
        },
        "model_id": {
            "type": "string",
            "description": "The model ID from Huggingface",
            "required": False,
        },
        "batch_size": {
            "type": "number",
            "default": 8,
            "description": "The batch size for the pipeline",
        },
        "item_key": {
            "type": "string",
            "description": "The key in the input item to use as input. Incoming values should be strings (or normal text)",
            "default": "",
            "required": True,
        },
        "device_id": {
            "type": "string",
            "default": "cuda",
            "description": 'The device ID (e.g, "cuda:0", "cpu") to use',
            "required": True,
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
        "parallelize_preprocessing": {
            "type": "boolean",
            "default": True,
            "description": "Whether to parallelize preprocessing by sending inputs to the worker pool",
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
        task: str,
        model_id: str,
        batch_size: int,
        item_key: str,
        device_id: str,
        fp16: bool,
        log_model_outputs: bool,
        parallelize_preprocessing: bool,
        kwargs: dict = {},
    ):
        if parallelize_preprocessing:
            self.load_fn = self._load_fn
        super().__init__(batch_size, item_key)
        if task == "":
            task = None
        if model_id == "":
            model_id = None
        self.pipe = pipeline(
            task=task,
            model=model_id,
            batch_size=batch_size,
            device=device_id,
            torch_dtype=torch.float16 if fp16 else None,
            **kwargs
        )
        self.log_model_outputs = log_model_outputs
        self.parallelize_preprocessing = parallelize_preprocessing
        preprocess_params, forward_params, postprocess_params = (
            self.pipe._sanitize_parameters()
        )

        self.preprocess_params = {**self.pipe._preprocess_params, **preprocess_params}
        self.forward_params = {**self.pipe._forward_params, **forward_params}
        self.postprocess_params = {
            **self.pipe._postprocess_params,
            **postprocess_params,
        }
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.collate_fn = (
            no_collate_fn
            if self.batch_size == 1
            else pad_collate_fn(self.pipe.tokenizer, self.pipe.feature_extractor)
        )

    @torch.no_grad()
    def _load_fn(self, item: str):
        data = self.pipe.preprocess(item, **self.preprocess_params)
        return data

    @torch.no_grad()
    def on_item_batch(self, inputs, items, notes):
        if not self.parallelize_preprocessing or inputs == None:
            inputs = [self._load_fn(i) for i in items]
        inputs = [self.collate_fn(inputs)]  # To make this an iterable

        model_iterator = PipelineIterator(
            inputs,
            self.pipe.forward,
            self.forward_params,
            loader_batch_size=self.batch_size,
        )
        final_iterator = PipelineIterator(
            model_iterator, self.pipe.postprocess, self.postprocess_params
        )

        for output, note in zip(final_iterator, notes):
            output: ModelOutput
            if self.log_model_outputs:
                self.log(output, "json")
            note["model_output"] = output


from typing import List
import graphbook.steps as steps
import torch
import torchvision.transforms.functional as F
from graphbook.utils import image
from .hf_pipeline import Pipeline

class NLPPipeline(Pipeline):
    Parameters = {
        "task": {
            "type": "string",
            "description": "A task name from the Huggingface pipeline library",
            "required": False,
        }
    } | Pipeline.Parameters
    def __init__(
        self,
        task: str,
        model_id: str,
        batch_size: int,
        item_key: str,
        device_id: str,
        fp16: bool,
        log_model_outputs: bool,
        parallelize_preprocessing: bool,
        kwargs: dict = {},
    ):
        super().__init__(task, model_id, batch_size, item_key, device_id, fp16, log_model_outputs, parallelize_preprocessing, kwargs)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
