from typing import List
import graphbook.steps as steps
import torch
from transformers import pipeline
from transformers.pipelines.base import pad_collate_fn, no_collate_fn
from transformers.utils.generic import ModelOutput
from transformers.pipelines.pt_utils import PipelineIterator
from .hf_pipeline import Pipeline

class AudioPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
        batch_size: int,
        item_key: str,
        device_id: str,
        fp16: bool,
        log_model_outputs: bool,
        parallelize_preprocessing: bool,
        kwargs: dict = {},
    ):
        super().__init__(model_id, batch_size, item_key, device_id, fp16, log_model_outputs, parallelize_preprocessing, kwargs)

    @torch.no_grad()
    def _load_fn(self, item: any):
        data = self.pipe.preprocess(item, **self.preprocess_params)
        data['input_values'] = data['input_values'].to(self.pipe.torch_dtype)
        return data
