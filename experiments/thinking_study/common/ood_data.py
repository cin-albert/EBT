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


def ood_texts(name: str) -> list[str]:
    from data.nlp.bigbench_dataloader import BigBenchDataset
    from data.nlp.gsm8k_dataloader import GSM8KDataset
    from data.nlp.squad_dataloader import SQuADDataset

    hp = Namespace(dataset_dir="", execution_mode="pretrain")
    make = {"gsm8k": lambda: GSM8KDataset(hp, "test"),
            "squad": lambda: SQuADDataset(hp, "validation"),
            "bigbench_elementary_math_qa": lambda: BigBenchDataset(hp, "validation", "elementary_math_qa"),
            "bigbench_dyck_languages": lambda: BigBenchDataset(hp, "validation", "dyck_languages")}
    ds = make[name]()
    return [ds[i] for i in range(len(ds))]


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
