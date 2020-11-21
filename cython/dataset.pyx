from cython.operator cimport dereference # noqa: see https://youtrack.jetbrains.com/issue/PY-9087

from config cimport CppConfig, get_config_for_reading_csv


cdef class Dataset:

    def  __cinit__(self):
        self.dataset = NULL

    def __dealoc__(self):
        if  self.dataset is not NULL:
            del self.dataset

    cdef CppDataset* get_dataset(self):
        return self.dataset

    @property
    def columns(self):
        return self.dataset.num_total_features()

    @property
    def rows(self):
        return self.dataset.num_data()

    @property
    def number_features(self):
        return self.dataset.num_total_features()

    @property
    def feature_names(self):
        feature_names = self.dataset.feature_names()
        # return [x.decode("UTF-8") for x in feature_names]
        return feature_names

    @property
    def label_index(self):
        return self.dataset.label_idx()

    @property
    def label_name(self):
        raise NotImplementedError()

    @staticmethod
    def read_from_parquet(file_path, label_name, categorical_feature_names=None):
        raise NotImplementedError("Reading parquet files is not supported yet.")

    @staticmethod
    cdef CppDatasetLoader* get_dataset_loader(const char* file_path, CppConfig* config):
        cdef CppDatasetLoader* dataset_loader = new CppDatasetLoader(
            dereference(config),
            NULL, # PredictFunction is only needed if using input initial model (continued train)
            1, # TODO `number_classes` find out why this argument is needed as it's not clear from the code
            file_path) # TODO `file_path` find out why this path would be needed here
        return dataset_loader


    @staticmethod
    def read_from_csv(file_path, label_name, categorical_feature_names=None, Dataset reference_dataset=None):
        file_path_ = file_path.encode("UTF-8") # noqa: PyCharm thinks file_path is the class identifier (self)
        cdef:
            Dataset py_dataset = Dataset()
            CppConfig* config = get_config_for_reading_csv(label_name, categorical_feature_names)
            CppDatasetLoader* dataset_loader = Dataset.get_dataset_loader(file_path_, config)

        if reference_dataset is None:
            py_dataset.dataset = dataset_loader.LoadFromFile(file_path_)
        elif isinstance(reference_dataset, Dataset):
            py_dataset.dataset = dataset_loader.LoadFromFileAlignWithOtherDataset(file_path_, reference_dataset.dataset)
        else:
            raise ValueError("Argument `reference_dataset` must be a Dataset class.")

        return py_dataset

    @staticmethod
    def read_training_set_from_csv(training_set_file_path, label_name, categorical_feature_names=None):
        return Dataset.read_from_csv(training_set_file_path, label_name, categorical_feature_names)

    @staticmethod
    def read_validation_set_from_csv(validation_set_file_path, label_name, categorical_feature_names=None, reference_dataset=None):
        if reference_dataset is None:
            raise ValueError("Argument `reference_dataset` must be defined when reading validation set.")
        return Dataset.read_from_csv(validation_set_file_path, label_name, categorical_feature_names, reference_dataset)
