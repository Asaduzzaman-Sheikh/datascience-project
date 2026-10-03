import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")

            # ── 1. Split features (X) and target (y) ──
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            # ── 2. Define candidate models ──
            models = {
                "Random Forest":       RandomForestRegressor(),
                "Decision Tree":       DecisionTreeRegressor(),
                "Gradient Boosting":   GradientBoostingRegressor(),
                "Linear Regression":   LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor":        XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":  AdaBoostRegressor(),
            }

            # ── 3. Define hyperparameter grids per model ──
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse',
                                'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample':     [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators':  [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators':  [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    'depth':         [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations':    [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators':  [8, 16, 32, 64, 128, 256],
                },
            }

            # ── 4. Evaluate all models (tune + test) ──
            model_report, tuned_models = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                models=models,   params=params
            )
            logging.info(f"Model report: {model_report}")

            # ── 5. Pick the best model (Pythonic way) ──
            best_model_name  = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model       = tuned_models[best_model_name]   # ← TUNED model!
            
            

            logging.info(f"Best model: {best_model_name} with R² = {best_model_score:.4f}")

            # ── 6. Threshold check ──
            if best_model_score < 0.6:
                raise CustomException(
                    Exception(f"No model scored above 0.6. Best: "
                        f"{best_model_name} ({best_model_score:.4f})"),
                    sys
                )

            # ── 7. Save the best (tuned) model ──
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model 
            )
            logging.info(f"Saved {best_model_name} to "
                        f"{self.model_trainer_config.trained_model_file_path}")

            # ── 8. Final R² of the saved model ──
            predicted  = best_model.predict(X_test)
            r2_square  = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)   
        
        

if __name__ == '__main__':
    from src.components.data_transformation import DataTransformation

    # 1. Load transformed data
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path="artifacts/train.csv",
            test_path="artifacts/test.csv"
        )

    print(f"\n📊 train_arr shape: {train_arr.shape}")
    print(f"📊 test_arr shape:  {test_arr.shape}")
    print(f"📦 preprocessor:    {preprocessor_path}")

    # 2. Train and evaluate models
    model_trainer = ModelTrainer()
    r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"\n" + "═" * 55)
    print(f"  🏆 FINAL R² = {r2_square:.4f}")
    print(f"  💾 Model saved to: {model_trainer.model_trainer_config.trained_model_file_path}")
    print("═" * 55 + "\n")
    