from cython.operator cimport dereference # noqa: see https://youtrack.jetbrains.com/issue/PY-9087
from libcpp.vector cimport vector

from config cimport CppConfig
from dataset cimport CppDataset

cdef vector[const CppMetric*] get_metrics(list metrics, CppDataset* training_data, CppConfig* config):

    cdef vector[const CppMetric*] cpp_metrics

    for metric in metrics:
        current_metric = CppMetric.CreateMetric(metric.encode("UTF-8"), dereference(config))
        if current_metric == NULL:
            # TODO check how to propagate errors
            raise ValueError(f"Cannot build metric: `{metric}`")
        current_metric.Init(training_data.metadata(), training_data.num_data())
        cpp_metrics.push_back(current_metric) # noqa (not detecting method from vector)

    return cpp_metrics
