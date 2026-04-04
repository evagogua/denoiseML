import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset 
from typing import List, Dict, Any, Optional

class ClassificationDataset(TorchDataset):

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        if 'denoise_labels' in data.column_names:
            data = data.remove_columns('denoise_labels')
        if 'classification_labels' in data.column_names:
            data = data.rename_column('classification_labels', 'labels')
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        unique_labels = set()
        for item in self.data:
            unique_labels.add(item['labels'])
        
        self.tags_ = sorted(list(unique_labels))
        
        self.label_to_id = {label: i for i, label in enumerate(self.tags_)}
        self.id_to_label = {i: label for i, label in enumerate(self.tags_)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        item = self.data[index]

        text = " ".join(item["text"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        label_str = item["labels"]
        if label_str not in self.label_to_id:
          raise ValueError(f"Unknown label: {label_str}")
        label_id = self.label_to_id[label_str]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

    def get_labels(self) -> List[str]:
        return self.tags_

    def get_label_mapping(self) -> Dict[str, int]:
        return self.label_to_id
        
def prepare_classification_dataset_from_json(data_json, tokenizer, test_size=0.2, seed=42, verbose=True):

    hf_dataset = Dataset.from_json(data_json)

    split_dataset = hf_dataset.train_test_split(test_size=test_size, seed=seed)

    train_dataset = ClassificationDataset(data=split_dataset["train"], tokenizer=tokenizer)
    test_dataset = ClassificationDataset(data=split_dataset["test"], tokenizer=tokenizer)

    if verbose:
        print("Пример из train (исходные данные):")
        print(split_dataset["train"][2])
        print("\nПример из train (после токенизации):")
        print(train_dataset[2])
        print()

    return train_dataset, test_dataset