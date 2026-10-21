"""The paper's OOD eval sets (§4.1.3, Table 3) as harness items.

Texts come from the repo loaders in their `pretrain` format ("Question: ...\nAnswer: ..."), the format
`get_ppl` scores, and are tokenized like the pretraining data (GPT-NeoX, truncated to SEQ_LEN + 1).
`n_prompt` counts the tokens up to and including "Answer:", for the secondary answer-only metric.
"""
from argparse import Namespace

from torch.utils.data import Dataset

from harness import SEQ_LEN

OOD_SETS = ("gsm8k", "squad", "bigbench_elementary_math_qa", "bigbench_dyck_languages")
ANSWER_SEP = "\nAnswer: "


PARQUET = {  # name -> (HF dataset repo, pinned revision, parquet path); the repo loaders' splits
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet"),
    "squad": ("rajpurkar/squad_v2", "3ffb306f725f7d2ce8394bc1873b24868140c412", "squad_v2/validation-00000-of-00001.parquet"),
    "bigbench_elementary_math_qa": ("tasksource/bigbench", "b8ee34693560a13a9feac6aa5540449e18b7eddb",
                                    "elementary_math_qa/validation-00000-of-00001.parquet"),
    "bigbench_dyck_languages": ("tasksource/bigbench", "b8ee34693560a13a9feac6aa5540449e18b7eddb",
                                "dyck_languages/validation-00000-of-00001.parquet"),
}


def loader_texts(loader_cls, rows) -> list[str]:
    """Format rows with a repo loader's own `pretrain` __getitem__, skipping its __init__ (Hub download)."""
    ds = loader_cls.__new__(loader_cls)
    ds.hparams, ds.dataset = Namespace(execution_mode="pretrain"), rows
    return [ds[i] for i in range(len(rows))]


def ood_texts(name: str) -> list[str]:
    """Texts of one OOD set. The loaders' own __init__ goes through the Hub module path, whose metadata
    datasets 3.0.0 can no longer parse (feature type "List"), so the parquet files are read directly."""
    from datasets import load_dataset

    from data.nlp import bigbench_dataloader, gsm8k_dataloader, squad_dataloader

    repo, rev, path = PARQUET[name]
    rows = load_dataset("parquet", data_files=f"hf://datasets/{repo}@{rev}/{path}", split="train")
    if name == "squad":
        rows = rows.filter(squad_dataloader.filter)  # answerable questions only, as SQuADDataset does
    cls = {"gsm8k": gsm8k_dataloader.GSM8KDataset, "squad": squad_dataloader.SQuADDataset}.get(
        name, bigbench_dataloader.BigBenchDataset)
    return loader_texts(cls, rows)


def encode(text: str, tokenizer) -> dict:
    prompt = text[: text.rindex(ANSWER_SEP) + len(ANSWER_SEP) - 1]  # ends with "Answer:"; the answer keeps its leading space
    return {"input_ids": tokenizer(text, truncation=True, max_length=SEQ_LEN + 1)["input_ids"],
            "n_prompt": len(tokenizer(prompt)["input_ids"])}


class OODDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer):
        self.texts, self.tokenizer = texts, tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return encode(self.texts[i], self.tokenizer)
