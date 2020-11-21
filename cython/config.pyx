from libcpp.unordered_map cimport unordered_map

from config cimport CppConfig


def get_supported_parameters():
    return [
        "boosting",
        "objective",
        "num_iterations",
        "learning_rate",
        "early_stopping_round",
        "random_seed"
    ]

def verify_parameters_or_fail(dict parameters):
    for parameter in parameters.keys():
        if parameter not in get_supported_parameters():
            raise ValueError(f"Parameter `{parameter}` not supported yet.")


cdef CppConfig* get_config_for_reading_csv(label_name, categorical_feature_names=None):
    label_name = label_name.encode("UTF-8")
    cdef CppConfig* config = new CppConfig()
    config.header = True
    config.label_column = b"name:" + label_name
    if categorical_feature_names is not None:
        categorical_feature_names_cpp = ("name:" + ",".join(categorical_feature_names)).encode("UTF-8")
        config.categorical_feature = categorical_feature_names_cpp
    return config


cdef CppConfig* get_config_for_binary_classification(const unordered_map[string, string] parameters):
    cdef CppConfig* config = new CppConfig()
    config.Set(parameters)
    # Hardcoded parameters for this specific case (classification)
    config.objective = b"binary"
    config.sigmoid = 1.0
    config.force_col_wise = True
    # TODO Hardcoded to false until better performance investigation
    config.boost_from_average = False
    return config


cdef CppConfig* get_config_for_regression(const unordered_map[string, string] parameters):
    cdef CppConfig* config = new CppConfig()
    config.Set(parameters)
    # Hardcoded parameters for this specific case (classification)
    config.objective = b"regression"
    config.force_col_wise = True
    # TODO Hardcoded to false until better performance investigation
    config.boost_from_average = False
    return config
