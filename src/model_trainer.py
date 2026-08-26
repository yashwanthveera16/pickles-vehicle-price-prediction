import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    """
    A reusable class for training and evaluating regression models.

    This class can be used with any Scikit-learn compatible regression model,
    allowing the same workflow to be reused across multiple experiments.
    """

    # Shared experiment log across all models
    experiment_log = []

    def __init__(self, model, model_name):
        # Store the model object and its name
        self.model = model
        self.model_name = model_name

    def train(self, X_train, y_train):
        """
        Train the selected regression model.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Generate predictions for the test dataset.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance using MAE, RMSE and R² Score.
        """

        # Generate predictions
        predictions = self.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Store evaluation metrics
        metrics = {
            "Model": self.model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R² Score": r2
        }

        # Log experiment results
        ModelTrainer.experiment_log.append(metrics)

        return predictions, metrics

    def print_metrics(self, metrics):
        """
        Display evaluation metrics in a readable format.
        """

        print(f"\n{metrics['Model']} Results")
        print("-" * 35)
        print(f"MAE      : {metrics['MAE']:.2f}")
        print(f"RMSE     : {metrics['RMSE']:.2f}")
        print(f"R² Score : {metrics['R² Score']:.4f}")

    @classmethod
    def get_experiment_log(cls):
        """
        Return the experiment log containing all model results.
        """
        return cls.experiment_log

    @classmethod
    def clear_experiment_log(cls):
        """
        Clear all previously logged experiments.
        """
        cls.experiment_log = []