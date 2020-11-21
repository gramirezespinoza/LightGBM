from cython.operator cimport dereference # noqa: see https://youtrack.jetbrains.com/issue/PY-9087
from libc.stdlib cimport free, malloc
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from numpy import genfromtxt, array

from config cimport CppConfig, get_config_for_binary_classification, get_config_for_regression
from config import verify_parameters_or_fail
from dataset cimport Dataset
from metric cimport CppMetric, get_metrics
from objective cimport CppObjectiveFunction, get_objective_function
from prediction cimport CppPredictionEarlyStopConfig, CreatePredictionEarlyStopInstance


cdef unordered_map[string, string] __get_parameter_map_from_dict(dict parameters):
    cdef unordered_map[string, string] parameter_map = unordered_map[string, string]()
    for k, v in parameters.items():
        parameter_map[k.encode("UTF-8")] = str(v).encode("UTF-8")
    return parameter_map


cdef Boosting __get_booster(Dataset training_data, list metrics, CppConfig* config):
    """

    Parameters
    ----------
    training_data: TODO
    metrics: list of metrics to evaluate training dataset. Not supported yet.
    config: TODO

    Returns
    -------

    """
    cdef:
        Boosting py_boosting = Boosting()
        CppBoosting* boosting
        CppObjectiveFunction* objective_function = get_objective_function(training_data.dataset, config)
        vector[const CppMetric*] cpp_metrics = get_metrics(metrics, training_data.dataset, config)

    # Filename set to NULL as there's no need to save the model
    boosting = CppBoosting.CreateBoosting(config.boosting, NULL)

    boosting.Init(config, training_data.get_dataset(), objective_function, cpp_metrics)
    py_boosting.boosting = boosting
    py_boosting.config = config
    return py_boosting


cdef class Boosting:

    cdef CppBoosting* boosting
    cdef CppConfig* config

    def  __cinit__(self):
        self.boosting = NULL
        self.config = NULL

    def __dealoc__(self):
        if  self.boosting is not NULL:
            del self.boosting
        if self.config is not NULL:
            del self.config

    def train_model(self, list validation_datasets=None, list validation_metrics=None):

        if validation_datasets is not None:
            if validation_metrics is None:
                # TODO exception message
                raise ValueError("TODO CHANGE ME METRICS NEEDED")

            for dataset in validation_datasets:

                tmp: Dataset = dataset
                cpp_metrics = get_metrics(validation_metrics, tmp.dataset, self.config)
                self.boosting.AddValidDataset(tmp.dataset, cpp_metrics)

        # Disable snapshot/output info to file
        snapshot_frequency = 0
        self.boosting.Train(snapshot_frequency, b"")

        return None

    def _predict_internal(self, csv_file_path, prediction_type):

        numpy_for_prediction = genfromtxt(csv_file_path, delimiter=",", dtype="double", skip_header=1)
        number_rows = numpy_for_prediction.shape[0]
        cdef:
            double [:, :] data_for_prediction = numpy_for_prediction
            double* prediction = <double *> malloc(number_rows * sizeof(double))
            CppPredictionEarlyStopConfig early_stopping_config

        # Note about prediction early stopping:
        #   * Why is this needed? Not explained clearly on the docs
        #   * Totally disabled in this cython version of LightGBM
        #   * An instance of CppPredictionEarlyStopConfig and CppPredictionEarlyStopInstance is required
        #     for the code to run properly but the "type" of CppPredictionEarlyStopInstance is set to "none"
        #     which deactivates the functionality
        early_stopping_for_prediction = CreatePredictionEarlyStopInstance(b"none", early_stopping_config)

        # Disabling predict_contrib for the moment
        self.boosting.InitPredict(0, self.config.early_stopping_round, False)

        if prediction_type == "raw":
            for i in range(number_rows):
                self.boosting.PredictRaw(&data_for_prediction[i, 0], &prediction[i], &early_stopping_for_prediction)
        elif prediction_type == "transformed":
            for i in range(number_rows):
                self.boosting.Predict(&data_for_prediction[i, 0], &prediction[i], &early_stopping_for_prediction)
        else:
            raise ValueError("Only two prediction types supported: raw and transformed.")

        prediction_python = array([e for e in prediction[:number_rows]])
        free(prediction)
        return prediction_python

    def predict(self, csv_file_path):
        return self._predict_internal(csv_file_path, prediction_type="transformed")

    def predict_raw(self, csv_file_path):
        return self._predict_internal(csv_file_path, prediction_type="raw")

    @property
    def feature_names(self):
        return self.boosting.FeatureNames()

    @property
    def number_of_classes(self):
        # TODO what is this number? Hard-coded to 1 in GBDT Class
        return self.boosting.NumberOfClasses()

    @staticmethod
    def get_booster_for_binary_classification(Dataset training_data, dict parameters):
        verify_parameters_or_fail(parameters)
        parameter_map = __get_parameter_map_from_dict(parameters)
        config = get_config_for_binary_classification(parameter_map)
        return __get_booster(training_data, [], config)

    @staticmethod
    def get_booster_for_regression(Dataset training_data, dict parameters):
        verify_parameters_or_fail(parameters)
        parameter_map = __get_parameter_map_from_dict(parameters)
        config = get_config_for_regression(parameter_map)
        return __get_booster(training_data, [], config)
