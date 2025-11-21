import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss


def top_k_accuracy(probs, y_true, classes, k=3):
    # probs: n_samples x n_classes
    correct = 0
    class_to_index = {c: i for i, c in enumerate(classes)}
    total = 0
    for i in range(len(y_true)):
        true = y_true.iloc[i]
        if true not in class_to_index:
            continue
        total += 1
        true_idx = class_to_index[true]
        topk = np.argsort(probs[i])[::-1][:k]
        if true_idx in topk:
            correct += 1
    return (correct / total) if total > 0 else 0.0


def main():
    print("Loading model and data...")
    model = pickle.load(open('model.pkl', 'rb'))
    try:
        meta = pickle.load(open('meta.pkl', 'rb'))
        columns = meta.get('columns')
    except Exception:
        columns = pickle.load(open('columns.pkl', 'rb'))
        meta = {'columns': columns}

    df = pd.read_csv('training_data.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.fillna(0)

    X = df.drop('prognosis', axis=1)
    y = df['prognosis']

    # use the same split as training script
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Running predictions on hold-out set...")
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)
        # create dummy probs: 1 for predicted class, 0 otherwise
        classes = getattr(model, 'classes_', np.unique(preds))
        probs = np.zeros((len(preds), len(classes)))
        class_to_index = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(preds):
            probs[i, class_to_index[p]] = 1.0

    print("Classification report (hold-out):")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix shape:", cm.shape)

    # log loss (lower is better)
    try:
        ll = log_loss(y_test, probs, labels=model.classes_)
        print(f"Log loss: {ll:.4f}")
    except Exception as e:
        print("Could not compute log loss:", e)

    # Top-k accuracy
    try:
        top1 = top_k_accuracy(probs, y_test, model.classes_, k=1)
        top3 = top_k_accuracy(probs, y_test, model.classes_, k=3)
        print(f"Top-1 accuracy: {top1*100:.2f}%")
        print(f"Top-3 accuracy: {top3*100:.2f}%")
    except Exception as e:
        print("Could not compute top-k accuracy:", e)

    # Print brief metrics from meta if present
    if meta.get('metrics'):
        print("Metrics saved in meta.pkl (from training):")
        for cls, stats in meta['metrics'].items():
            if isinstance(stats, dict) and 'precision' in stats:
                print(f"{cls}: precision={stats['precision']:.2f}, recall={stats['recall']:.2f}, f1={stats['f1-score']:.2f}")

    print("Done.")


if __name__ == '__main__':
    main()
