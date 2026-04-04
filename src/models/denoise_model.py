import numpy as np
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset

def make_last_subtoken_mask(mask, has_cls=True, has_eos=True):
    if has_cls:
        mask = mask[1:]
    if has_eos:
        mask = mask[:-1]
    is_last_word = list((first != second) for first, second in zip(mask[:-1], mask[1:])) + [True]
    if has_cls:
        is_last_word = [False] + is_last_word
    if has_eos:
        is_last_word.append(False)
    return is_last_word

class DenoiseDataset(TorchDataset):

    def __init__(self, data, tokenizer, min_count=3, tags=None):
        if 'denoise_labels' in data.column_names:
            data = data.rename_column('denoise_labels', 'labels')
        if 'classification_labels' in data.column_names:
            data = data.remove_columns('classification_labels')
        self.data = data
        self.tokenizer = tokenizer
        self.tags_ = ['0', 'N']
        self.tag_indexes_ = {tag: i for i, tag in enumerate(self.tags_)}
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokenization = self.tokenizer(item["text"], is_split_into_words=True)
        last_subtoken_mask = make_last_subtoken_mask(tokenization.word_ids())
        answer = {"input_ids": tokenization["input_ids"], "mask": last_subtoken_mask}
        if "labels" in item:
            labels = [self.tag_indexes_.get(tag) for tag in item["labels"]]
            zero_labels = np.array([self.ignore_index] * len(tokenization["input_ids"]), dtype=int)
            zero_labels[last_subtoken_mask] = labels
            answer["labels"] = zero_labels

        return answer

def prepare_denoise_dataset_from_json(data_json, tokenizer, test_size=0.2, seed=42, verbose=True):

    hf_dataset = Dataset.from_json(data_json)

    split_dataset = hf_dataset.train_test_split(test_size=test_size, seed=seed)

    train_dataset = DenoiseDataset(data=split_dataset["train"], tokenizer=tokenizer)
    test_dataset = DenoiseDataset(data=split_dataset["test"], tokenizer=tokenizer)

    if verbose:
        print("Пример из train (исходные данные):")
        print(split_dataset["train"][2])
        print("\nПример из train (после токенизации):")
        print(train_dataset[2])
        print()

    return train_dataset, test_dataset

