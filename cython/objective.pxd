from libcpp.string cimport string

from config cimport CppConfig
from dataset cimport CppMetadata, CppDataset


cdef extern from "LightGBM/objective_function.h" namespace "LightGBM" nogil:

    ctypedef long data_size_t

    cdef cppclass CppObjectiveFunction "LightGBM::ObjectiveFunction":

        @staticmethod
        CppObjectiveFunction* CreateObjectiveFunction(const string type, const CppConfig& config) # noqa (naming convention)

        void Init(const CppMetadata& metadata, data_size_t num_data) # noqa (naming convention)


cdef CppObjectiveFunction* get_objective_function(CppDataset* training_dataset, CppConfig* config)
