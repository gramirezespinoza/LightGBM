import pytest

from boosting import Boosting
from dataset import Dataset


@pytest.fixture()
def random_seed_for_training():
    return  666


@pytest.fixture()
def training_dataset():
    return Dataset.read_training_set_from_csv(
        training_set_file_path="csv_data/credit_card_prediction.csv",
        label_name="Class")


def test_early_stopping_with_validation(random_seed_for_training):
    parameters = {
        "objective": "binary_logloss",
        "num_iterations": 100,
        "boosting": "gbdt",
        "early_stopping_round": 5,
        "random_seed": random_seed_for_training,
    }

    training_dataset = Dataset.read_training_set_from_csv(
        training_set_file_path="csv_data/credit_card_prediction.csv",
        label_name="Class")

    validation_dataset = Dataset.read_validation_set_from_csv(
        validation_set_file_path="csv_data/credit_card_validation.csv",
        label_name="Class",
        reference_dataset=training_dataset)

    (Boosting
        .get_booster_for_binary_classification(training_dataset, parameters)
        .train_model(
            validation_datasets=[validation_dataset, ],
            validation_metrics=["binary_logloss", "auc"]))
