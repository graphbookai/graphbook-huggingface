from typing import List
import graphbook.core.steps as steps
import torch
import torchvision.transforms.functional as F
from graphbook.core.utils import image

class MergeMasks(steps.Step):
    """
    Merges multiple masks into a single mask by summing them and thresholding the result.
    
    Args:
        output_key (str): The key in the output item to use as output
        delete_raw_output (bool): Whether to delete the raw Huggingface model output from the item
    """
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

    def on_data(self, data):
        if data[self.output_key] is None:
            data[self.output_key] = []
        output = data["model_output"]
        tensors = []
        for o in output:
            tensors.append(F.to_tensor(o["mask"]))
        if len(tensors) != 0:
            output = torch.stack(tensors)
            output = torch.sum(output, dim=0)
            output = torch.where(output > 0, 1.0, 0.0)
            output = F.to_pil_image(output)
            data[self.output_key] = {"type": "image", "value": output}

        if self.delete_raw_output:
            del data["model_output"]

class FilterMasks(steps.Step):
    """
    Filters the masks based on the labels provided.
    
    Args:
        labels (List[str]): The labels to filter for
    """
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

    def on_data(self, data):
        output = data["model_output"]
        filtered_output = [o for o in output if o["label"] in self.labels]
        data["model_output"] = filtered_output

class MaskOutputs(steps.Step):
    """
    Parses the model output as masks and converts them to images for display in the Graphbook UI.
    
    Args:
        output_key (str): The key in the output item to use as output
        delete_raw_output (bool): Whether to delete the raw Huggingface model output
    """
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

    def on_data(self, data):
        output = data["model_output"]
        for o in output:
            o["type"] = "image"
            o["value"] = o["mask"]
            del o["mask"]
        data[self.output_key] = output

        if self.delete_raw_output:
            del data["model_output"]


class DepthOutputs(steps.Step):
    """
    Parses the model output as depth maps and converts them to images for display in the Graphbook UI.
    
    Args:
        output_key (str): The key in the output item to use as output
    """
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

    def on_data(self, data):
        output = image(data["model_output"]["depth"])
        del data["model_output"]
        data[self.output_key] = output


class ImageClassificationMaxLabel(steps.Step):
    """
    Outputs the label with the maximum score from the model output.
    
    Args:
        delete_raw_hf_output (bool): Whether to delete the raw Huggingface model output
    """
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

    def on_data(self, data):
        output = data["model_output"]
        max_label = ""
        max_score = float("-inf")
        for out in output:
            if out["score"] > max_score:
                max_score = out["score"]
                max_label = out["label"]
        data["prediction"] = max_label
        if self.delete_raw_hf_output:
            del data["model_output"]
