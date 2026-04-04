import numpy as np
import scipy.special
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from src.data.denoise_model import DenoiseDataset

def denoising(data_samples, tr, tokenizer, tag):
  biganswer = []
  test_dataset = DenoiseDataset(data_samples, tokenizer)
  predictions = predict_with_trainer(tr, test_dataset, classes=test_dataset.tags_)
  for item, probs in zip(data_samples, predictions):
    new_sample = {"text":[token for token, mask in zip(item['text'], probs['labels']) if mask == '0'], 'denoise_labels': item["denoise_labels"], 'classification_labels': item["classification_labels"]}
    masked = [item for item in predictions[0]["labels"] if item != -100]
    biganswer.append(new_sample)
  return biganswer
  
def predict_with_trainer(trainer, dataset, classes):
    predictions = trainer.predict(dataset)
    answer = []
    for elem, curr_predictions in zip(dataset, predictions.predictions):
        mask = elem["mask"]
        probs = scipy.special.softmax(curr_predictions, axis=-1)[:len(mask)]
        best_indexes = np.argmax(probs, axis=-1)[elem["mask"]]
        best_labels = np.take(classes, best_indexes)
        best_probs = np.max(probs, axis=-1)[elem["mask"]]
        curr_answer = {"labels": best_labels, "probs": best_probs}
        answer.append(curr_answer)
    return answer
    
def evaluate_on_test_set(trainer, test_samples, tokenizer, classes=['0', 'N']):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    from datasets import Dataset

    testing = DenoiseDataset(data=Dataset.from_list(test_samples), tokenizer=tokenizer)

    predictions = predict_with_trainer(trainer, testing, classes=classes)

    all_true = []
    all_pred = []

    for word, label in zip(test_samples, predictions):
        print(" ".join(word['text']) + '\n' +
              " ".join(label['labels']) + '\n' +
              " ".join(word['labels']))

        correct_count = sum(1 for i, k in zip(label['labels'], word['labels']) if i == k)
        print(f"{correct_count}/{len(word['labels'])}")

        true_numeric = [0 if x == '0' else 1 for x in word['labels']]
        pred_numeric = [0 if x == '0' else 1 for x in label['labels']]
        all_true.extend(true_numeric)
        all_pred.extend(pred_numeric)

    print()


    accuracy = accuracy_score(all_true, all_pred)

    precision_0 = precision_score(all_true, all_pred, pos_label=0, zero_division=0)
    recall_0 = recall_score(all_true, all_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(all_true, all_pred, pos_label=0, zero_division=0)

    precision_1 = precision_score(all_true, all_pred, pos_label=1, zero_division=0)
    recall_1 = recall_score(all_true, all_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(all_true, all_pred, pos_label=1, zero_division=0)

    macro_f1 = (f1_0 + f1_1) / 2

    print("="*50)
    print("ОСНОВНЫЕ МЕТРИКИ")
    print("="*50)
    print(f"Token Accuracy:    {accuracy:.2%}")
    print()
    print(f"КЛАСС 0 (ЦЕЛЕВАЯ РЕЧЬ):")
    print(f"  Precision: {precision_0:.2%}")
    print(f"  Recall:    {recall_0:.2%}")
    print(f"  F1:        {f1_0:.2%}")
    print()
    print(f"КЛАСС N (ШУМ):")
    print(f"  Precision: {precision_1:.2%}")
    print(f"  Recall:    {recall_1:.2%}")
    print(f"  F1:        {f1_1:.2%}")
    print()
    print(f"Macro F1: {macro_f1:.2%}")
    print("="*50 + "\n")

test = [
  {
    "text": "переведите деньги на мой счет".split(),
    "labels": ['0','0','0','0','0']
  },
  {
    "text": "алиса какой канал переведите сто долларов на сберкнижку".split(),
    "labels": ['N','N','N','0','0','0','0','0']
  },
  {
    "text": "выключи телевизор заблокируй карту срочно пожалуйста".split(),
    "labels": ['N','N','0','0','0','0']
  },
  {
    "text": "привет как дела что делаешь переведи пять тысяч на вклад".split(),
    "labels": ['N','N','N','N','N','0','0','0','0','0']
  },
  {
    "text": "ну я ему говорю а он такой слушай переведите пожалуйста средства".split(),
    "labels": ['N','N','N','N','N','N','N','N','0','0','0']
  },
  {
    "text": "какой баланс моей карты".split(),
    "labels": ['0','0','0','0']
  },
  {
    "text": "погода сегодня алиса какой баланс на карте".split(),
    "labels": ['N','N','N','0','0','0','0']
  },
  {
    "text": "переведи деньги срочно сделай тише я сказал".split(),
    "labels": ['0','0','0','N','N','N','N']
  },
  {
    "text": "перевод средств на карту сбербанка".split(),
    "labels": ['0','0','0','0','0']
  },
  {
    "text": "иди сюда быстрее посмотри что я нашел переведи сто баксов".split(),
    "labels": ['N','N','N','N','N', 'N', 'N', '0', '0', '0']
  }
]
