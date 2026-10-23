import pytest
from transformers import AutoTokenizer

import harness
import ood_data


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=False)
    t.pad_token_id = t.eos_token_id
    return t


def test_answer_tokens_start_after_prompt(tok):
    item = ood_data.encode("Question: What is 2+3?\nAnswer: 5", tok)
    assert tok.decode(item["input_ids"][item["n_prompt"]:]) == " 5"


def test_prompt_tokens_are_a_prefix_of_the_full_text(tok):
    text = "Question: Complete the sequence: [ ( < >\nAnswer: ) ]"
    item = ood_data.encode(text, tok)
    prompt = tok(text[: text.rindex(ood_data.ANSWER_SEP) + len(ood_data.ANSWER_SEP) - 1])["input_ids"]
    assert item["input_ids"][: item["n_prompt"]] == prompt


def test_long_text_is_truncated_and_answer_falls_outside(tok):
    item = ood_data.encode("Question: " + "word " * 600 + "\nAnswer: 7", tok)
    assert len(item["input_ids"]) == harness.SEQ_LEN + 1
    assert item["n_prompt"] > harness.SEQ_LEN + 1


def test_answer_masks_mark_targets_from_the_first_answer_token():
    ds = [{"input_ids": [1, 2, 3, 4, 5], "n_prompt": 3}, {"input_ids": [1, 2, 3]}]
    m = next(harness.answer_masks(ds, [0, 1], 2))
    assert m.shape == (2, harness.SEQ_LEN)
    assert m[0, :4].tolist() == [False, False, True, True]  # targets 2, 3 predict tokens 3, 4 (the answer)
    assert not m[1].any()


def test_eval_dataset_rp_val_is_the_validation_split(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(harness, "validation_split", lambda hp: sentinel)
    assert harness.eval_dataset("rp_val", {}) is sentinel


def test_eval_dataset_rejects_unknown_names():
    with pytest.raises(ValueError):
        harness.eval_dataset("mmlu", {"tokenizer": "EleutherAI/gpt-neox-20b"})


def test_rows_are_formatted_by_the_repo_loaders_pretrain_format():
    from data.nlp.bigbench_dataloader import BigBenchDataset
    from data.nlp.gsm8k_dataloader import GSM8KDataset

    assert ood_data.loader_texts(GSM8KDataset, [{"question": "q?", "answer": "42"}]) == ["Question: q?\nAnswer: 42"]
    rows = [{"inputs": "Close the brackets: ( [ Answer:", "targets": ["] )"]}]
    assert ood_data.loader_texts(BigBenchDataset, rows) == ["Question: Close the brackets: ( [\nAnswer: ] )"]


def test_subset_of_a_short_set_uses_every_item_once():
    idx = harness.eval_subset_indices(5, 1024)
    assert sorted(idx) == [0, 1, 2, 3, 4]
