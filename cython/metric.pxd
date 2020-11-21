from libcpp.string cimport string
from libcpp.vector cimport vector

from config cimport CppConfig
from dataset cimport CppDataset, CppMetadata


cdef extern from "LightGBM/metric.h" namespace "LightGBM" nogil:

    ctypedef long data_size_t

    cdef cppclass CppMetric "LightGBM::Metric":

        void Init(const CppMetadata& metadata, data_size_t num_data)

        @staticmethod
        CppMetric* CreateMetric(const string type, const CppConfig& config) # noqa

cdef vector[const CppMetric*] get_metrics(list metrics, CppDataset* training_data, CppConfig* config)
