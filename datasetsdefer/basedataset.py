from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Abstract method for learning to defer methods"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """must at least have data_dir, test_split, val_split, batch_size, transforms"""
        pass

    @abstractmethod
    def generate_data(self):
        """generates the data loader, called on init
        
        should generate the following must:
            self.data_train_loader
            self.data_val_loader
            self.data_test_loader
            self.d (dimension)
            self.n_dataset (number of classes in target)
        """
        pass
