from .nodes import HuggingfacePipeline, HuggingfaceDataset, AssignModelOutputsToNotes
import os.path as osp
from graphbook.plugins import export, web


export("HuggingfacePipeline", HuggingfacePipeline)
export("HuggingfaceDataset", HuggingfaceDataset)
export("AssignModelOutputsToNotes", AssignModelOutputsToNotes)

web(osp.realpath(osp.join(osp.dirname(__file__), "./dist/bundle.js")))
