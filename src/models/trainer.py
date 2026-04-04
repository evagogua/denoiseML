import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from src.models.denoise_model import DenoiseDataset

def plot_training_curves(trainer):

    train_losses = []
    eval_losses = []
    lrs = []
    steps = []

    for log in trainer.state.log_history:
        if 'loss' in log and 'eval_loss' not in log:
            train_losses.append(log['loss'])
            steps.append(log.get('step', len(train_losses)))

        if 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])

        if 'learning_rate' in log:
            lrs.append(log['learning_rate'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps[:len(train_losses)], train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if eval_losses:
        eval_steps = [log['step'] for log in trainer.state.log_history if 'eval_loss' in log]
        axes[1].plot(eval_steps, eval_losses, 'r-', linewidth=2, label='Eval Loss')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Evaluation Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    axes[2].plot(steps[:len(lrs)], lrs, 'g-', linewidth=2)
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    pred_labels = np.argmax(logits, axis=-1)

    all_pred = []
    all_true = []

    for i, (pred_sent, true_sent) in enumerate(zip(pred_labels, labels)):
        mask = true_sent != -100
        pred_filtered = pred_sent[mask]
        true_filtered = true_sent[mask]

        all_pred.extend(pred_filtered)
        all_true.extend(true_filtered)

    accuracy = accuracy_score(all_true, all_pred)

    precision_0 = precision_score(all_true, all_pred, pos_label=0, zero_division=0)
    recall_0 = recall_score(all_true, all_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(all_true, all_pred, pos_label=0, zero_division=0)

    precision_1 = precision_score(all_true, all_pred, pos_label=1, zero_division=0)
    recall_1 = recall_score(all_true, all_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(all_true, all_pred, pos_label=1, zero_division=0)

    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    return {
        "Accuracy": 100 * accuracy,

        "Precision_0": 100 * precision_0,
        "Recall_0": 100 * recall_0,
        "F1_0": 100 * f1_0,

        "Precision_N": 100 * precision_1,
        "Recall_N": 100 * recall_1,
        "F1_N": 100 * f1_1,

        "Macro_Precision": 100 * macro_precision,
        "Macro_Recall": 100 * macro_recall,
        "Macro_F1": 100 * macro_f1,
    }


def final_report(trainer, dataset):

    preds = np.argmax(trainer.predict(dataset).predictions, axis=-1)

    all_true, all_pred = [], []
    for i in range(len(dataset)):
        true = dataset[i]['labels']
        pred = preds[i][:len(true)]
        for t, p in zip(true, pred):
            if t != -100:
                all_true.append(t)
                all_pred.append(p)

    print(classification_report(all_true, all_pred,
                                target_names=['Целевая речь', 'Шум'],
                                zero_division=0))

