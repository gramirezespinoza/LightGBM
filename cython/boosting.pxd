from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

from config cimport CppConfig
from dataset cimport CppDataset, Dataset
from metric cimport CppMetric
from objective cimport CppObjectiveFunction
from prediction cimport CppPredictionEarlyStopInstance

cdef extern from "LightGBM/boosting.h" namespace "LightGBM" nogil:

    ctypedef double score_t
    ctypedef long int64_t

    cdef cppclass CppBoosting "LightGBM::Boosting":

        void Init(const CppConfig* config, # noqa (naming convention)
                  const CppDataset* train_data,
                  const CppObjectiveFunction* objective_function,
                  const vector[const CppMetric*] training_metrics)

        vector[string] FeatureNames() # noqa
        int NumberOfClasses() # noqa

        void Train(int snapshot_freq, const string model_output_path) # noqa
        void AddValidDataset(const CppDataset* valid_data, const vector[const CppMetric*]& valid_metrics) # noqa
        void InitPredict(int start_iteration, int num_iteration, bool is_pred_contrib) # noqa
        void Predict(const double* features, double* output, const CppPredictionEarlyStopInstance* early_stop) # noqa
        void PredictRaw(const double* features, double* output, const CppPredictionEarlyStopInstance* early_stop) # noqa

        @staticmethod # noqa pycharm does not recognize static method decorator for cython
        CppBoosting* CreateBoosting(const string type, const char* filename) # noqa


cdef unordered_map[string, string] __get_parameter_map_from_dict(dict parameters)
cdef __get_booster(Dataset training_data, list metrics, CppConfig* config)
