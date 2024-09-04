import numpy as np
from skimage.exposure import rescale_intensity
from torch import from_numpy, clip

class SelectRandomTimepoint(object):
    """Select a random timepoint form the time series

    time_dimension: int
        dimension index of time 
    """

    def __init__(self, time_dimension):
        self.td = time_dimension

    def __call__(self, sample):
        shape = sample.shape
        random_tp = np.random.randint(0,shape[self.td])
        
        slice_objects = [
            random_tp if i == self.td else slice(0,shape[i]) for i in range(len(shape))
        ]
        return sample[slice_objects]
    
class SelectSpecificTimepoint(object):
    """Select a random timepoint form the time series

    time_dimension: int
        dimension index of time 
    """

    def __init__(self, time_dimension, timepoint):
        self.td = time_dimension
        self.timepoint = timepoint

    def __call__(self, sample):
        shape = sample.shape
        
        slice_objects = [
            self.timepoint if i == self.td else slice(0,shape[i]) for i in range(len(shape))
        ]
        return sample[slice_objects]

class SelectRandomTPNumpy(object):
    """Select a random timepoint form the time series

    time_dimension: int
        dimension index of time 
    """

    def __init__(self, time_dimension):
        self.td = time_dimension
        
    def __call__(self, sample):
        shape = sample.shape
        random_tp = np.random.randint(0,shape[self.td])
        
        out = np.take(sample,[random_tp],axis=self.td).squeeze(self.td)
        # print(out.shape)
        return out
    
class SelectSpecificTPNumpy(object):
    """Select a random timepoint form the time series

    time_dimension: int
        dimension index of time 
    """

    def __init__(self, time_dimension, timepoint):
        self.td = time_dimension
        self.timepoint = timepoint

    def __call__(self, sample):
        out = np.take(sample,[self.timepoint],axis=self.td).squeeze(self.td)
        return out

class CustomToTensor(object):
    """Custom ToTensor: works with any shape and does not normalize
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        return from_numpy(sample).float()
    
class ShiftIntensity(object):
    """ShiftIntensity: shift intensity by float factor and clip within (0, 1)
    """

    def __init__(self, bf_factor):
        self.bf_factor = np.random.uniform(0.5, bf_factor)

    def __call__(self, sample):
        return clip((sample * self.bf_factor), 0, 1)