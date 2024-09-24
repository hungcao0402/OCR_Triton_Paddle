import numpy as np


class BaseRecLabelDecode:
    def __init__(self, character_dict_path=None, use_space_char=True):
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        self.character = self.add_special_char(dict_character )
        self.dict = dict(enumerate(self.character))


    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        for batch_idx in range(len(text_index)):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list) if conf_list else np.nan))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=True):
        super().__init__(character_dict_path)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, remove_duplicate=True)

        rec_texts, rec_scores = [], []
        for res in text:
            rec_texts.append(res[0])
            rec_scores.append(res[1])
        return rec_texts, rec_scores


    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
