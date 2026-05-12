import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from training_data import X_test, X_train, params, y_test, y_train

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MLflow POC")

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="iris-logistic-regression"):
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    