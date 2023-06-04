import pickle
import numpy as np
import yaml
import json
from hyperopt import hp, fmin, tpe, space_eval
from loguru import logger
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


logger.info('TRAINING MODULE')
params = yaml.safe_load(open('params.yaml'))['training']

train_data_path = params.get('train_data_path', 'train_data.pkl')
model_name = params.get('model_name', 'model.pkl')
hyperparams = params.get('hyperparams', None)
logs_path = params.get('logs_path', None)

logger.info('Loading prepared data...')
with open(logs_path + train_data_path, 'rb') as fd:
    matrix = pickle.load(fd)

y_train = np.squeeze(matrix[:, 0].toarray())
x_train = matrix[:, 1:]

logger.info('Model creating and fitting...')
clf = LGBMClassifier(random_state=42)
clf.fit(x_train, y_train)

logger.info('Saving trained model...')
with open(logs_path + model_name + '.pkl', "wb") as fd:
    pickle.dump(clf, fd)

space = {k: eval(v) for k, v in hyperparams.items()}

num_folds = params.get('num_folds', 3)
max_evals = params.get('max_evals', 5)


def objective(params):
    lgbm = LGBMClassifier(**params)
    cv = StratifiedKFold(num_folds)
    score = cross_val_score(lgbm, x_train, y_train,
                            scoring='roc_auc',
                            cv=cv,
                            n_jobs=-1).mean()
    return 1 - score

logger.info('Hyperparameters tuning...')
best = fmin(objective, space,
            algo=tpe.suggest,
            max_evals=max_evals)

best_params = space_eval(space, best)

best_params = {i: int(v) if type(v) not in [float, int] else v for i, v in
               best_params.items()}

print(best_params)

logger.info('Saving best hyperparameters...')
with open(logs_path + 'best_params.json', 'w') as fp:
    json.dump(best_params, fp)

logger.info('Model fitting with best hyperparameters...')
clf_tuned = LGBMClassifier(**best_params)
clf_tuned.fit(x_train, y_train)

logger.info('Saving trained with best hyperparameters model...s')
with open(logs_path + model_name + '_tuned' + '.pkl', "wb") as fd:
    pickle.dump(clf_tuned, fd)
