from typing import List
import graphbook.steps as steps
import torch
import torchvision.transforms.functional as F
from graphbook.utils import image

class MergeMasks(steps.Step):
    RequiresInput = True
    Parameters = {
        "output_key": {
            "type": "string",
            "description": "The key in the output item to use as output",
            "required": True,
        },
        "delete_raw_output": {
            "type": "boolean",
            "default": True,
            "description": "Whether to delete the raw Huggingface model output from the item",
        },
    }
    Outputs = ["out"]
    Category = "Huggingface/Post Processing"

    def __init__(self, output_key: str, delete_raw_output: bool):
        super().__init__()
        self.output_key = output_key
        self.delete_raw_output = delete_raw_output

    def on_note(self, note):
        if note[self.output_key] is None:
            note[self.output_key] = []
        output = note["model_output"]
        tensors = []
        for o in output:
            tensors.append(F.to_tensor(o["mask"]))
        if len(tensors) != 0:
            output = torch.stack(tensors)
            output = torch.sum(output, dim=0)
            output = torch.where(output > 0, 1.0, 0.0)
            output = F.to_pil_image(output)
            note[self.output_key] = {"type": "image", "value": output}

        if self.delete_raw_output:
            del note["model_output"]

class FilterMasks(steps.Step):
    RequiresInput = True
    Parameters = {
        "labels": {
            "type": "list[string]",
            "description": "The labels to filter for",
            "required": True,
        },
    }
    Outputs = ["out"]
    Category = "Huggingface/Post Processing"

    def __init__(self, labels: List[str]):
        super().__init__()
        self.labels = labels

    def on_note(self, note):
        output = note["model_output"]
        filtered_output = [o for o in output if o["label"] in self.labels]
        note["model_output"] = filtered_output

class MaskOutputs(steps.Step):
    RequiresInput = True
    Parameters = {
        "output_key": {
            "type": "string",
            "description": "The key in the output item to use as output",
            "required": True,
        },
        "delete_raw_output": {
            "type": "boolean",
            "default": True,
            "description": "Whether to delete the raw Huggingface model output from the item",
        },
    }
    Outputs = ["out"]
    Category = "Huggingface/Post Processing"

    def __init__(self, output_key: str, delete_raw_output: bool):
        super().__init__()
        self.output_key = output_key
        self.delete_raw_output = delete_raw_output

    def on_note(self, note):
        output = note["model_output"]
        for o in output:
            o["type"] = "image"
            o["value"] = o["mask"]
            del o["mask"]
        note[self.output_key] = output

        if self.delete_raw_output:
            del note["model_output"]


class DepthOutputs(steps.Step):
    RequiresInput = True
    Parameters = {
        "output_key": {
            "type": "string",
            "description": "The key in the output item to use as output",
            "required": True,
        },
    }
    Outputs = ["out"]
    Category = "Huggingface/Post Processing"

    def __init__(self, output_key: str):
        super().__init__()
        self.output_key = output_key

    def on_note(self, note):
        output = image(note["model_output"]["depth"])
        del note["model_output"]
        note[self.output_key] = output


class ImageClassificationMaxLabel(steps.Step):
    RequiresInput = True
    Parameters = {
        "delete_raw_hf_output": {
            "type": "boolean",
            "default": True,
            "description": "Whether to delete the raw Huggingface model output from the item",
        },
    }
    Outputs = ["out"]
    Category = "Huggingface/Post Processing"

    def __init__(self, delete_raw_hf_output: bool):
        super().__init__()
        self.delete_raw_hf_output = delete_raw_hf_output

    def on_note(self, note):
        output = note["model_output"]
        max_label = ""
        max_score = float("-inf")
        for out in output:
            if out["score"] > max_score:
                max_score = out["score"]
                max_label = out["label"]
        note["prediction"] = max_label
        if self.delete_raw_hf_output:
            del note["model_output"]
