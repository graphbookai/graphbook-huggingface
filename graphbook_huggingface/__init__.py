from . import hf_postprocessing as pp
from . import hf_pipeline

from .hf_datasets import HuggingfaceDataset
import os.path as osp
from graphbook.plugins import export, web


export("TransformersPipeline", hf_pipeline.TransformersPipeline)
export("MergeMasks", pp.MergeMasks)
export("FilterMasks", pp.FilterMasks)
export("MaskOutputs", pp.MaskOutputs)
export("DepthOutputs", pp.DepthOutputs)
export("ImageClassificationMaxLabel", pp.ImageClassificationMaxLabel)

export("HuggingfaceDataset", HuggingfaceDataset)

web(osp.realpath(osp.join(osp.dirname(__file__), "./dist/bundle.js")))
