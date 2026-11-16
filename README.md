# InvoiceRL: fine-tuning a small LLM with verifiable rewards

This project teaches a small open model (**Qwen2.5-0.5B-Instruct**) to extract invoice data as JSON. Training uses reinforcement learning (GRPO), where a **program checks every answer**. There's no judge model: the reward is computed directly from the ground truth.

## Why
Many real AI-agent tasks (bills, invoices, contracts) have answers a program can verify. Using a deterministic checker as the reward signal teaches the model to be *correct*, not just *plausible*.

## Two ways to run it

### 1. Colab notebook: full pipeline on a free T4 GPU
[`notebooks/InvoiceRL_colab.ipynb`](notebooks/InvoiceRL_colab.ipynb)

1. **Synthetic data:** 1,500 training and 200 held-out test invoices with exact ground truth, in 3 messy layouts with varied date and number formats.
2. **Baseline:** the base model, zero-shot.
3. **LoRA SFT:** r=16 adapters on the attention and MLP projections, 1 epoch.
4. **GRPO:** starts from the SFT model, 4 generations per prompt, with a reward that gives:
   - 0.2 for valid JSON
   - 0.1 for each correct field (vendor, invoice_no, date, subtotal, tax, total)
   - 0.2 when the line items, subtotal, tax and total are arithmetically consistent (a calculator check)
5. **Evaluation:** every stage is scored on the same held-out set, followed by error analysis.

Open it in Colab, choose Runtime → Change runtime type → **T4 GPU**, then run all cells (about 1.5–2.5 hours).

### 2. Python scripts: GRPO only, for a local GPU
- [`invoicerl.py`](invoicerl.py): synthetic invoices and the reward. The reward gives 0.2 for valid JSON plus 0.8 × the fraction of 4 fields correct (vendor, invoice_number, date, total). Run `python invoicerl.py` for the self-check.
- [`train.py`](train.py): GRPO training with TRL. It reports the evaluation reward before training and every 100 steps.

```bash
python3 -m venv .venv && .venv/bin/pip install trl datasets
SMOKE=1 .venv/bin/python train.py   # quick pipeline check: 135M model, 3 steps, CPU
.venv/bin/python train.py           # full run: CUDA GPU with ~16GB
```

## Results
<!-- Paste the table from section 6 of the notebook (results.md). Report the numbers exactly as measured. -->

| Model | JSON valid % | Field accuracy % | All fields correct % | Arithmetic consistent % |
|---|---|---|---|---|
| Base (zero-shot) | _pending_ | | | |
| LoRA SFT | _pending_ | | | |
| LoRA SFT + GRPO | _pending_ | | | |

## Error analysis
<!-- From notebook section 7: what the final model still gets wrong (date formats, commas in amounts, missed line items, rounding). -->
_Pending first full run._

## Limitations
- The data is synthetic. Real invoices (scans, OCR noise, multiple pages) are harder.
- The model is small (0.5B) and GRPO runs briefly on free hardware, so the results show the method rather than a production model.

**Stack:** Python, PyTorch, Hugging Face Transformers, TRL (SFTTrainer, GRPOTrainer), PEFT (LoRA), Datasets.
