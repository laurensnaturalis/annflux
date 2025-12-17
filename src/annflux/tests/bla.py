from numpy._typing import NDArray
import numpy as np


class SomeClass(object):
    features_: NDArray[np.float64]

    def __init__(self):
        self.loaded = False

    def load_data(self):
        self.features_ = np.zeros((1,))
        self.loaded = True

    @property
    def features(self) -> NDArray[np.float64]:
        return self.features_


a = SomeClass()
if not a.loaded:
    a.load_data()
else:
    if len(a.features) > 0:
        # do something with a.features
        pass

