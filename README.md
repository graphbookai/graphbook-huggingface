<p align="center">
  <a href="https://graphbook.ai">
    <img src="assets/graphbook-hf-banner.png" alt="Logo" width=512>
  </a>

  <h1 align="center">Graphbook Hugging Face</h1>

  <p align="center">
    Build No Code Hugging Face AI Pipelines
  </p>
</p>

You can build efficient DAG workflows or AI pipelines without any code. This is a Graphbook plugin that lets you drag and drop Hugging Face models and datasets onto Graphbook workflows. This plugin contains a web panel for searching and drag-and-dropping models and datasets from [Huggingface Hub](https://huggingface.co/) onto their graphbook workflows.

<img src="assets/example-hf-pipeline.png" alt="Example Pipeline with Hugging Face" with=1024>

## Packaged Nodes

Graphbook Hugging Face contains the following nodes:

* `TransformersPipeline` step for model usage from transformers package
* `HuggingfaceDataset` step for dataset usage from the datasets package
* And numerous `Post Processing/*` steps for post processing of model outputs

## Getting started
1. `pip install graphbook_huggingface graphbook transformers datasets`
1. `graphbook --config hf.config.yaml`

