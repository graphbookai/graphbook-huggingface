from .nodes import HuggingfacePipeline, AssignModelOutputToNotes
import os.path as osp
from graphbook.plugins import export, web


export("HuggingfacePipeline", HuggingfacePipeline)
export("AssignModelOutputToNotes", AssignModelOutputToNotes)

web(osp.realpath(osp.join(osp.dirname(__file__), "../web/dist/bundle.js")))
