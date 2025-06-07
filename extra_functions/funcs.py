from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def model_evaluation(model, X_train, y_train, X_valid, y_valid):
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    scores = {
        "train_mse": mean_squared_error(y_train, train_pred),
        "train_mae": mean_absolute_error(y_train, train_pred),
        "train_r2": r2_score(y_train, train_pred),
        "valid_mse": mean_squared_error(y_valid, valid_pred),
        "valid_mae": mean_absolute_error(y_valid, valid_pred),
        "valid_r2": r2_score(y_valid, valid_pred)
    }
    return scores