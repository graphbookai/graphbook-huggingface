from typing import List
import graphbook.steps as steps
import torch
import torchvision.transforms.functional as F
from graphbook.utils import image

class MergeMasks(steps.Step):
    RequiresInput = True
    Parameters = {
        "item_key": {
            "type": "string",
            "description": "The key in the input item to use as input",
            "required": True,
        },
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

    def __init__(self, item_key: str, output_key: str, delete_raw_output: bool):
        super().__init__(item_key)
        self.output_key = output_key
        self.delete_raw_output = delete_raw_output

    def on_item(self, item, note):
        if note[self.output_key] is None:
            note[self.output_key] = []
        output = item["model_output"]
        tensors = []
        for o in output:
            tensors.append(F.to_tensor(o["mask"]))
        if len(tensors) != 0:
            output = torch.stack(tensors)
            output = torch.sum(output, dim=0)
            output = torch.where(output > 0, 1.0, 0.0)
            output = F.to_pil_image(output)
            note[self.output_key].append({"type": "image", "value": output})

        if self.delete_raw_output:
            del item["model_output"]

class FilterMasks(steps.Step):
    RequiresInput = True
    Parameters = {
        "item_key": {
            "type": "string",
            "description": "The key in the input item to use as input",
            "required": True,
        },
        "labels": {
            "type": "list[string]",
            "description": "The labels to filter for",
            "required": True,
        },
    }
    Outputs = ["out"]
    Category = "Huggingface/Post Processing"

    def __init__(self, item_key: str, labels: List[str]):
        super().__init__(item_key)
        self.labels = labels

    def on_item(self, item, *_):
        output = item["model_output"]
        filtered_output = [o for o in output if o["label"] in self.labels]
        item["model_output"] = filtered_output

class MaskOutputs(steps.Step):
    RequiresInput = True
    Parameters = {
        "item_key": {
            "type": "string",
            "description": "The key in the input item to use as input",
            "required": True,
        },
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

    def __init__(self, item_key: str, output_key: str, delete_raw_output: bool):
        super().__init__(item_key)
        self.output_key = output_key
        self.delete_raw_output = delete_raw_output

    def on_item(self, item, note):
        output = item["model_output"]
        for o in output:
            o["type"] = "image"
            o["value"] = o["mask"]
            del o["mask"]
        note[self.output_key] = output

        if self.delete_raw_output:
            del item["model_output"]


class DepthOutputs(steps.Step):
    RequiresInput = True
    Parameters = {
        "item_key": {
            "type": "string",
            "description": "The key in the input item to use as input",
            "required": True,
        },
        "output_key": {
            "type": "string",
            "description": "The key in the output item to use as output",
            "required": True,
        },
    }
    Outputs = ["out"]
    Category = "Huggingface/Post Processing"

    def __init__(self, item_key: str, output_key: str):
        super().__init__(item_key)
        self.output_key = output_key

    def on_item(self, item, note):
        output = image(item["model_output"]["depth"])
        del item["model_output"]
        note[self.output_key] = output


class ImageClassificationMaxLabel(steps.Step):
    RequiresInput = True
    Parameters = {
        "item_key": {
            "type": "string",
            "description": "The key in the input item to use as input",
            "required": True,
        },
        "delete_raw_hf_output": {
            "type": "boolean",
            "default": True,
            "description": "Whether to delete the raw Huggingface model output from the item",
        },
    }
    Outputs = ["out"]
    Category = "Huggingface/Post Processing"

    def __init__(self, item_key: str, delete_raw_hf_output: bool):
        super().__init__(item_key)
        self.delete_raw_hf_output = delete_raw_hf_output

    def on_item(self, item, *_):
        output = item["model_output"]
        max_label = ""
        max_score = float("-inf")
        for out in output:
            if out["score"] > max_score:
                max_score = out["score"]
                max_label = out["label"]
        item["prediction"] = max_label
        if self.delete_raw_hf_output:
            del item["model_output"]
