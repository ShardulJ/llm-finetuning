# Medical QA — Fine-tuning Mistral-7B on PubMedQA

Fine-tuning Mistral-7B-Instruct on PubMedQA using QLoRA to answer biomedical research questions. Given a research question and a supporting abstract, the model outputs yes, no, or maybe with a brief explanation grounded in the provided context.

---

## Results

| Model | Accuracy (10 samples) |
|---|---|
| Random baseline | 33.3% |
| Mistral-7B-Instruct + QLoRA | 50.0% |

Full predictions are in `results/eval_results.json`.

The model handles clear yes/no questions well. It struggles on maybe cases, which make up a smaller portion of PubMedQA and are genuinely ambiguous even for humans. A few failures were also caused by truncation — some abstracts exceeded the 512 token limit and got cut mid-sentence, which broke the response format.

---

## Example outputs

**Question:** Does implant coating with antibacterial-loaded hydrogel reduce bacterial colonization in vitro?

**Model output:**
```
YES: DAC hydrogel is able to deliver antibacterial compounds in vitro.
Antibacterial-loaded DAC hydrogel is able to reduce bacterial colonization
and biofilm formation on implants. Intraoperative coating is feasible and
resistant to press-fit insertion.
```

**Question:** Can APC mutation analysis contribute to therapeutic decisions in familial adenomatous polyposis?

**Model output (truncation failure):**
```
ations at the 5' end of codon 168 or the 3' end of codon 1580.
```
Label was `maybe` — the abstract was truncated mid-sentence at 512 tokens, causing the model to generate from an incomplete context.

---

## Setup

**Model:** `unsloth/mistral-7b-instruct-v0.2-bnb-4bit`  
**Dataset:** PubMedQA `pqa_labeled` — 1,000 labeled Q&A pairs, split 800/100/100  
**Method:** QLoRA — base weights quantized to 4-bit, LoRA adapters trained on q and v attention projections  
**Hardware:** Google Colab T4 (free tier), ~28 minutes training time

### LoRA config

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha (α) | 16 |
| Target modules | q_proj, v_proj |
| Dropout | 0.05 |

### Training config

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 1 (effective 4 with grad accumulation) |
| Learning rate | 2e-4 with cosine decay |
| Max sequence length | 512 tokens |

---

## Running the notebook

1. Open `notebooks/medical_finetuning.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Run Cell 1, restart runtime when prompted, then run remaining cells in order

---

## Loading the model

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "ShardulJ/mistral-pubmedqa-qlora",
    max_seq_length = 512,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

prompt = """<s>[INST] You are a biomedical research assistant. Given a research question and relevant abstract context, answer with 'yes', 'no', or 'maybe', followed by a concise explanation grounded in the provided context.

Question: Does aspirin reduce the risk of colorectal cancer?

Context:
A large cohort study followed 80,000 participants over 10 years.
Regular aspirin use was associated with a 19% reduction in colorectal cancer incidence (HR 0.81, 95% CI 0.71-0.93).
The effect was strongest for tumors in the proximal colon.
[/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## What I would do differently

Increasing `MAX_SEQ_LEN` to 1024 or 2048 would fix the truncation failures and likely push accuracy above 60%. That requires either Colab Pro (A100) or reducing the model size. Training on the full 1,000 samples without filtering would also help since the current filter removes some of the harder, longer examples that happen to include most of the maybe cases.

---

## References

- Jin et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. https://arxiv.org/abs/1909.06146
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. https://arxiv.org/abs/2305.14314
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Unsloth. https://github.com/unslothai/unsloth

---

## License

MIT. This adapter is intended for research purposes only.
