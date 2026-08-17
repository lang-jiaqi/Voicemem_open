# Voicemem Qwen3.6-35B-A3B QLoRA v2

This directory is the release manifest for the **adapter-only** checkpoint at training step 3318. The model weights are published separately on Hugging Face; they are intentionally not committed to this code repository.

- Hugging Face: [LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2](https://huggingface.co/LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2)
- Base model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- Adapter type: LoRA / PEFT, rank 32, alpha 64
- Checkpoint: `checkpoint-3318`, 2 epochs, 3318 steps
- Intended use: research on long-term conversational memory and memory-grounded answer generation.

## Install and load

Install a compatible `transformers`, `peft`, and the dependencies required by the upstream base model. Download the adapter from Hugging Face, then load it onto the permitted base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen3.6-35B-A3B"
adapter_id = "LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_id)
```

Do not treat this adapter as a standalone model. Users must obtain the base model under its own license and access conditions.

## What is released

The Hugging Face model repository should contain only:

- `adapter_model.safetensors`
- `adapter_config.json` with a public base-model identifier
- a completed Hugging Face Model Card

Optimizer shards, DeepSpeed state, RNG state, scheduler state, and training logs are not model artifacts and must not be uploaded as model weights.

## Evaluation

Evaluation protocol and recorded baseline are in [eval/README.md](eval/README.md). The trained-model score must be inserted only from the completed, archived AudioMC result file.

## Limitations

This is a research adapter. It has not been evaluated for safety-critical use, does not replace the base model's safety documentation, and may reproduce biases or errors from both training data and the base model.

## Release checklist

- [ ] Verify the base model license permits adapter redistribution and the stated use.
- [ ] Complete the Model Card with authors, data provenance, data licenses, and a citation.
- [ ] Insert the final trained-model AudioMC score and link the archived raw result.
- [ ] Create a Git tag and archive the release on Zenodo for a DOI.
