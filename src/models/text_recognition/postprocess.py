from __future__ import annotations

"""Local postprocessing for text recognition outputs."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.models.text_recognition.config import RecognitionPostprocessConfig


class RecognitionPostprocessor:
    """Decode recognition logits into strings and confidence scores."""

    def __init__(self, config: RecognitionPostprocessConfig) -> None:
        dictionary_path = str(config.dictionary_path) if config.dictionary_path else None
        self._decoder = CTCLabelDecode(dictionary_path, use_space_char=config.use_space_char)

    def __call__(self, preds: np.ndarray) -> Tuple[List[str], List[float]]:
        texts, scores = self._decoder(preds)
        return texts, scores


class BaseRecLabelDecode:
    def __init__(self, character_dict_path: str | None = None, use_space_char: bool = True):
        self.character_str: list[str] = []

        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
        else:
            with Path(character_dict_path).open("rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    self.character_str.append(line.decode("utf-8").strip("\n").strip("\r\n"))

        if use_space_char and " " not in self.character_str:
            self.character_str.append(" ")
        self.character = self.add_special_char(self.character_str)
        self.dict = dict(enumerate(self.character))

    def add_special_char(self, dict_character: list[str]) -> list[str]:
        return dict_character

    def decode(self, text_index, text_prob=None, remove_duplicate=False):
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        for batch_idx in range(len(text_index)):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if remove_duplicate and idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                    continue

                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, float(np.mean(conf_list)) if conf_list else float("nan")))
        return result_list

    @staticmethod
    def get_ignored_tokens():
        return [0]


class CTCLabelDecode(BaseRecLabelDecode):
    def __init__(self, character_dict_path: str | None = None, use_space_char: bool = True):
        super().__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, remove_duplicate=True)

        rec_texts: List[str] = []
        rec_scores: List[float] = []
        for res in text:
            rec_texts.append(res[0])
            rec_scores.append(res[1])
        return rec_texts, rec_scores

    def add_special_char(self, dict_character: list[str]) -> list[str]:
        return ["blank"] + dict_character
