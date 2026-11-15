"""GRPO fine-tune of a small instruct model on synthetic invoices.
python3 -m venv .venv && .venv/bin/pip install trl datasets   # own env: needs NumPy 2, unlike base Anaconda
.venv/bin/python train.py            # full run needs a CUDA GPU with ~16GB
SMOKE=1 .venv/bin/python train.py    # quick pipeline check on a Mac: 135M model, 3 tiny steps, no eval
"""
import os

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from invoicerl import invoice_reward, make_dataset

SMOKE = os.environ.get("SMOKE") == "1"
group = 4 if SMOKE else 8

trainer = GRPOTrainer(
    # 0.5B + Adam in fp32 is ~8GB, too much for small Macs; smoke uses a 135M model to test the pipeline
    model="HuggingFaceTB/SmolLM2-135M-Instruct" if SMOKE else "Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=invoice_reward,
    train_dataset=Dataset.from_list(make_dataset(16 if SMOKE else 2000, seed=0)),
    eval_dataset=None if SMOKE else Dataset.from_list(make_dataset(200, seed=1)),  # different seed = held-out invoices
    args=GRPOConfig(
        output_dir="invoicerl-out",
        num_generations=group,
        per_device_train_batch_size=group,
        max_completion_length=128,
        learning_rate=1e-6,
        logging_steps=1 if SMOKE else 10,
        max_steps=3 if SMOKE else -1,
        use_cpu=SMOKE,  # MPS on 8GB Macs runs out of memory after a step or two
        eval_strategy="no" if SMOKE else "steps",
        eval_steps=100,
    ),
)
if not SMOKE:
    trainer.evaluate()  # baseline reward before any training
trainer.train()
trainer.save_model()
