from libcpp.string cimport string
from libcpp.vector cimport vector

from config cimport CppConfig


cdef extern from "LightGBM/dataset.h" namespace "LightGBM" nogil:
    cdef cppclass CppMetadata "LightGBM::Metadata":
        pass


cdef extern from "LightGBM/dataset.h" namespace "LightGBM" nogil:
    ctypedef long data_size_t
    cdef cppclass CppDataset "LightGBM::Dataset":
        int label_idx() const
        int num_features() const
        int num_feature_groups() const # TODO What is a feature group?
        int num_total_features() const
        data_size_t num_data() const
        const vector[string] feature_names() const
        const CppMetadata& metadata() const


cdef extern from "LightGBM/dataset_loader.h" namespace "LightGBM" nogil:
    ctypedef void* PredictFunction
    cdef cppclass CppDatasetLoader "LightGBM::DatasetLoader":
        CppDatasetLoader(const CppConfig&, const PredictFunction&, int, const char*) # noqa (naming convention)
        CppDataset* LoadFromFile(const char *) # noqa (naming convention)
        CppDataset* LoadFromFileAlignWithOtherDataset(const char* filename, const CppDataset* train_data) # noqa


cdef class Dataset:
    cdef CppDataset* dataset
    cdef CppDataset* get_dataset(self)

    @staticmethod
    cdef CppDatasetLoader* get_dataset_loader(const char* file_path, CppConfig* config)
