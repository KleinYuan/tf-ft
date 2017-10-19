import numpy as np


class Augmentation:

    def __init__(self):
        self.X = None
        self.augmented_X = None
        self.augmented = False

    def load_data(self, X):
        self.X = X

    def augment_data(self):
        assert not self.augmented, '[Error] Data has already been augmented!'
        self.augmented_X = []
        for ind, img in enumerate(self.X):
            x = img.astype(np.uint8)
            augmented_x = self._augment_data(x)
            self.augmented_X.append(augmented_x)
        self.augmented_X = np.array(self.augmented_X)
        self.augmented_X = self.augmented_X.astype(np.float32)
        self.augmented = True

    def get_augmented_data(self):
        assert self.augmented, '[Error] Data was not augmented correctly!'
        return self.augmented_X

    @staticmethod
    def _augment_data(x):
        print 'To be delegated!'
        return x