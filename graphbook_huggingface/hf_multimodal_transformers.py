import torch
from .hf_audio_transformers import AudioPipeline

class MultimodalPipeline(AudioPipeline):

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
