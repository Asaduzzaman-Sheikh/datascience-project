import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    """Saves a Python object to disk using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Loads a pickled object from disk."""
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params,
                    n_iter=10, cv=5, random_state=42):
    """
    Tunes each model with RandomizedSearchCV, evaluates on the test set
    using R2, and returns:
        report:        {model_name: r2_score}
        tuned_models:  {model_name: best_tuned_estimator}
    """
    try:
        report = {}
        tuned_models = {}

        # PER-MODEL LOOP 
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            # ── Skip tuning if no params defined ──
            if params.get(model_name) is None or len(params[model_name]) == 0:
                logging.info(f"No hyperparameters for {model_name}. Skipping tuning.")
                model.fit(X_train, y_train)
                best_estimator = model
                best_params = {}
            else:
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params[model_name],
                    n_iter=n_iter,
                    scoring='r2',
                    cv=cv,
                    n_jobs=-1,
                    verbose=0,
                    random_state=random_state,
                )
                random_search.fit(X_train, y_train)
                best_estimator = random_search.best_estimator_
                best_params = random_search.best_params_

            # ── Evaluation runs for BOTH branches ──
            y_pred = best_estimator.predict(X_test)
            score = r2_score(y_test, y_pred)     

            report[model_name] = score
            tuned_models[model_name] = best_estimator

            logging.info(
                f"Model: {model_name}, R2 = {score:.4f}, Best Params: {best_params}"
            )

        
        logging.info(f"Model report: {report}")
        return report, tuned_models

    except Exception as e:
        raise CustomException(e, sys)