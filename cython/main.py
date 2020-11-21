# cython: language_level = 3
from numpy import sum as np_sum, genfromtxt as np_genfromtxt, logical_and
from boosting import Boosting
from dataset import Dataset

# TODO BE CAREFUL
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":

    train_dataset = Dataset.read_training_set_from_csv(
        training_set_file_path="csv_data/credit_card_train.csv",
        label_name="Class")

    validation_dataset_1 = Dataset.read_validation_set_from_csv(
        validation_set_file_path="csv_data/credit_card_validation.csv",
        label_name="Class",
        reference_dataset=train_dataset)

    parameters = {
        "objective": "binary_logloss",
        "num_iterations": 1000,
        "boosting": "gbdt",
        "early_stopping_round": 5,
        "random_seed": 666,
    }

    booster = Boosting.get_booster_for_binary_classification(train_dataset, parameters)

    booster.train_model(
        validation_datasets=[validation_dataset_1,],
        validation_metrics=["binary_logloss", "auc"])

    prediction_file = "/Users/gire/Desktop/Repositories/LightGBM/cython/csv_data/credit_card_prediction.csv"
    prediction = booster.predict(prediction_file)
    prediction_raw = booster.predict_raw(prediction_file)
    numpy_for_prediction = np_genfromtxt(prediction_file, delimiter=",", dtype="double", skip_header=1)

    print(f"Positives in prediction file: {np_sum(numpy_for_prediction[:, -1])}")
    print(f"All predicted positive: {np_sum(prediction >= 0.5)}")
    print(f"True positives: {np_sum(logical_and(prediction >= 0.5, numpy_for_prediction[:, -1] >= 0.5))}")

    print(f"Prediction array: {prediction}")
    print(f"Prediction array raw: {prediction_raw}")


