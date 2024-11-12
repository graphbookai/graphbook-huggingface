from . import hf_vision_transformers as v
from . import hf_nlp_transformers as n
from . import hf_audio_transformers as a
from . import hf_multimodal_transformers as m

from .hf_datasets import HuggingfaceDataset
import os.path as osp
from graphbook.plugins import export, web


export("VisionPipeline", v.VisionPipeline)
export("NLPPipeline", n.NLPPipeline)
export("AudioPipeline", a.AudioPipeline)
export("MultimodalPipeline", m.MultimodalPipeline)
export("MergeMasks", v.MergeMasks)
export("FilterMasks", v.FilterMasks)
export("MaskOutputs", v.MaskOutputs)
export("DepthOutputs", v.DepthOutputs)
export("ImageClassificationMaxLabel", v.ImageClassificationMaxLabel)

export("HuggingfaceDataset", HuggingfaceDataset)

web(osp.realpath(osp.join(osp.dirname(__file__), "./dist/bundle.js")))
