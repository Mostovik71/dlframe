import json
import yaml
import pickle
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

params = yaml.safe_load(open('params.yaml'))['evaluating']
params_train = yaml.safe_load(open('params.yaml'))['training']

def evaluate(model, matrix, filename):
    labels = matrix[:, 0].toarray().astype(int)
    x = matrix[:, 1:]

    probas = model.predict_proba(x)[:, 1]
    predictions = model.predict(x)
    
    metrics = {"roc_auc": roc_auc_score(labels, probas),
                 "precision_macro": precision_score(labels, predictions, average='macro'),
                 "recall_macro": recall_score(labels, predictions, average='macro'),
                 "f1_macro": f1_score(labels, predictions, average='macro')
                }
    with open(filename, 'w') as fp:
        json.dump(metrics, fp, indent=4)

    return metrics

model_path = params.get('model_path')
model_tuned_path = params.get('model_tuned_path')
train_data_path = params_train.get('train_data_path')
test_data_path = params.get('test_data_path')

with open(model_path, "rb") as fd:
    model = pickle.load(fd)

with open(model_tuned_path, "rb") as fd:
    model_tuned = pickle.load(fd)

with open(train_data_path, "rb") as fd:
    train = pickle.load(fd)

with open(test_data_path, "rb") as fd:
    test = pickle.load(fd)


train_metrics_without_tuning = evaluate(model, train, "train_metrics_without_tuning.json")
test_metrics_without_tuning = evaluate(model, test, "test_metrics_without_tuning.json")
train_metrics_with_tuning = evaluate(model_tuned, train, "train_metrics_with_tuning.json")
test_metrics_with_tuning = evaluate(model_tuned, test, "test_metrics_with_tuning.json")

metrics_before_tuning = pd.DataFrame([train_metrics_without_tuning, test_metrics_without_tuning], index=['train', 'test']).T
metrics_after_tuning = pd.DataFrame([train_metrics_with_tuning, test_metrics_with_tuning], index=['train', 'test']).T
metrics_difference = metrics_after_tuning - metrics_before_tuning

metrics_before_tuning.to_csv('metrics_before_tuning.csv', sep=';')
metrics_after_tuning.to_csv('metrics_after_tuning.csv', sep=';')
metrics_difference.to_csv('metrics_difference.csv', sep=';')

