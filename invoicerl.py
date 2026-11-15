"""InvoiceRL: synthetic invoices + a reward that checks the model's extracted fields.
Run `python invoicerl.py` for the self-check."""
import json
import random
import re
from datetime import date, timedelta

VENDORS = ["Acme Supplies Ltd", "Northwind Traders", "Globex Corp", "Initech LLC", "Umbrella Logistics", "Stark Office Co"]
ITEMS = ["Printer paper", "Toner cartridge", "Desk chair", "USB-C cable", "Consulting hour", "Monitor stand", "Shipping"]
PROMPT = ("Extract vendor, invoice_number, date (YYYY-MM-DD) and total (a number) from this invoice. "
          "Reply with only a JSON object with those four keys.\n\n")


def make_invoice(rng):
    d = date(2025, 1, 1) + timedelta(days=rng.randrange(700))
    lines = [(rng.choice(ITEMS), rng.randint(1, 9), round(rng.uniform(2, 400), 2)) for _ in range(rng.randint(1, 5))]
    subtotal = round(sum(q * p for _, q, p in lines), 2)
    tax = round(subtotal * rng.choice([0, 0.05, 0.1, 0.18]), 2)
    answer = {"vendor": rng.choice(VENDORS), "invoice_number": f"INV-{rng.randint(1000, 99999)}",
              "date": d.isoformat(), "total": round(subtotal + tax, 2)}
    # Varied date formats so the model has to normalise, not just copy.
    shown_date = rng.choice([d.isoformat(), d.strftime("%d/%m/%Y"), d.strftime("%B %d, %Y")])
    body = "\n".join(f"{n} x{q} @ ${p:,.2f} = ${q * p:,.2f}" for n, q, p in lines)
    text = (f"{answer['vendor']}\nInvoice #: {answer['invoice_number']}\nDate: {shown_date}\n\n{body}\n\n"
            f"Subtotal: ${subtotal:,.2f}\nTax: ${tax:,.2f}\nTotal due: ${answer['total']:,.2f}")
    return text, answer


def make_dataset(n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        text, answer = make_invoice(rng)
        # answer stored as a JSON string so the HF Dataset schema stays simple
        rows.append({"prompt": [{"role": "user", "content": PROMPT + text}], "answer": json.dumps(answer)})
    return rows


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def score(completion, answer):
    """0 for no parseable JSON object, else 0.2 for valid JSON + 0.8 * fraction of the 4 fields correct."""
    m = re.search(r"\{.*\}", completion, re.S)
    try:
        pred = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        pred = None
    if not isinstance(pred, dict):
        return 0.0
    total = _num(pred.get("total"))
    hits = [
        str(pred.get("vendor", "")).strip().lower() == answer["vendor"].lower(),
        str(pred.get("invoice_number", "")).strip() == answer["invoice_number"],
        str(pred.get("date", "")).strip() == answer["date"],
        total is not None and abs(total - answer["total"]) < 0.01,
    ]
    return 0.2 + 0.8 * sum(hits) / len(hits)


def invoice_reward(completions, answer, **kwargs):
    """TRL reward function: dataset column `answer` arrives as a kwarg."""
    texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
    return [score(t, json.loads(a)) for t, a in zip(texts, answer)]


if __name__ == "__main__":
    text, ans = make_invoice(random.Random(0))
    print(text, "\n")
    good = json.dumps(ans)
    assert score(good, ans) == 1.0
    assert score(f"```json\n{good}\n```", ans) == 1.0
    assert score(json.dumps({**ans, "total": f"${ans['total']:,.2f}"}), ans) == 1.0
    assert abs(score(json.dumps({**ans, "total": ans["total"] + 1}), ans) - 0.8) < 1e-9
    assert score(json.dumps({}), ans) == 0.2
    assert score("I think the total is 42", ans) == 0.0
    assert score("{not json}", ans) == 0.0
    rows = make_dataset(3, seed=0)
    assert invoice_reward([[{"role": "assistant", "content": r["answer"]}] for r in rows],
                          [r["answer"] for r in rows]) == [1.0, 1.0, 1.0]
    print("self-check ok")
