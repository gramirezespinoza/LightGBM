from cython.operator cimport dereference # noqa: see https://youtrack.jetbrains.com/issue/PY-9087

from config cimport CppConfig
from dataset cimport CppDataset

cdef CppObjectiveFunction* get_objective_function(CppDataset* training_dataset, CppConfig* config):
    cdef CppObjectiveFunction* objective_function = CppObjectiveFunction.CreateObjectiveFunction(
        config.objective,
        dereference(config))
    objective_function.Init(training_dataset.metadata(), training_dataset.num_data())
    return objective_function
