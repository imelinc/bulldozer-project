from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def rmse(y_true, y_pred):
    """Calculate the root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    """
        Evaluate the model using various metrics.
    """
    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)
    scores = {
        "Training MAE": mean_absolute_error(y_train, train_preds),
        "Validation MAE": mean_absolute_error(y_valid, valid_preds),
        "Training RMSE": rmse(y_train, train_preds),
        "Validation RMSE": rmse(y_valid, valid_preds),
        "Training R^2": r2_score(y_train, train_preds),
        "Validation R^2": r2_score(y_valid, valid_preds)
    }
    return scores