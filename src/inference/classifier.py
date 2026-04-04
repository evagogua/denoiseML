import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.special import softmax


def predict_with_trainer_seq(trainer, dataset, classes):
    predictions = trainer.predict(dataset)
    logits = predictions.predictions

    probs = softmax(logits, axis=-1)
    pred_ids = np.argmax(probs, axis=-1)

    pred_labels = [classes[i] for i in pred_ids]
    pred_probs = np.max(probs, axis=-1)

    return pred_labels, pred_probs


def get_simple_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division=0
    )

    print(f"Accuracy:  {accuracy*100}")
    print(f"Precision: {precision*100}")
    print(f"Recall:    {recall*100}")
    print(f"F1-score:  {f1*100}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_metrics_simple(eval_pred):

    predictions, labels = eval_pred
    predicted_labels = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted_labels, average='macro', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }