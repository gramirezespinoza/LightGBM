from libcpp.string cimport string

cdef extern from "LightGBM/prediction_early_stop.h" namespace "LightGBM" nogil:
    struct CppPredictionEarlyStopInstance "LightGBM::PredictionEarlyStopInstance":
        pass

    struct CppPredictionEarlyStopConfig "LightGBM::PredictionEarlyStopConfig":
        int round_period
        double margin_threshold

    CppPredictionEarlyStopInstance CreatePredictionEarlyStopInstance(const string& type,
                                                                     const CppPredictionEarlyStopConfig& config)
