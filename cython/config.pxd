from libcpp.string cimport string
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map


cdef extern from "LightGBM/config.h" namespace "LightGBM" nogil:
    cdef cppclass CppConfig "LightGBM::Config":

        # Dataset related
        bool header
        string label_column
        string categorical_feature

        # Boosting related
        string boosting
        string objective
        int num_iterations
        double learning_rate
        int early_stopping_round

        # Hardcoded or Not Supported
        double sigmoid
        bool boost_from_average
        bool force_col_wise

        # Prediction related
        bool pred_early_stop
        int pred_early_stop_freq
        double pred_early_stop_margin

        void Set(const unordered_map[string, string]& params) # noqa (naming convention)

cdef CppConfig* get_config_for_reading_csv(label_name, categorical_feature_names=?)
cdef CppConfig* get_config_for_binary_classification(const unordered_map[string, string] parameters)
cdef CppConfig* get_config_for_regression(const unordered_map[string, string] parameters)
