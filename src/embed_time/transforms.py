import numpy as np
from skimage.exposure import rescale_intensity
from torch import from_numpy
def rescale_bf(img,quantiles = [0.01,0.99]):
    min_max = np.quantile(img,quantiles)
    rescaled = (
        rescale_intensity(
            img,
            in_range=(min_max[0],min_max[1]),
            out_range=(0,1)) -1
    ) * -1
    rescaled = np.clip(rescaled,0,1)
    return rescaled

def rescale_bra(bra_tl,quantiles = [0.001,0.999]):
    min_max = np.quantile(bra_tl,quantiles)
    rescaled = rescale_intensity(
        bra_tl,
        in_range=(min_max[0],min_max[1]),
        out_range=(0,1)
    )
    return rescaled

def complex_normalisation(
        input_series, 
        bf_quant, 
        bra_quant
    ):
    """
    input_series: np.ndarray 
        dimensions = time, channel, y, x
    bf_quant: list
        lower and upper quantiles for rescaling brightfield images (channel 0)
        Performed for each image individually
    bra_quant: list
        lower and upper quantiles for rescaling brachyury images (channel 1)
        rescaled across the timelapse
    """
    bf_tl = input_series[:,0,:,:]
    bra_tl = input_series[:,1,:,:]
    out_bf = np.expand_dims(np.array([rescale_bf(img,bf_quant) for img in bf_tl]),1)
    out_bra = np.expand_dims(rescale_bra(bra_tl,bra_quant),1)
    return np.concatenate((out_bf,out_bra),axis=1)



class NormalizeCustom(object):
    """Normalise live TLS data with dimesnions t, c, y, x

    Args:
        bf_quantiles: list
            lower and upper quantiles for rescaling brightfield images (channel 0)
            Performed for each image individually
        bra_quantiles: list
            lower and upper quantiles for rescaling brachyury images (channel 1)
            rescaled across the timelapse
    """

    def __init__(self, bf_quantiles, bra_quantiles):
        self.bf_quantiles = bf_quantiles
        self.bra_quantiles = bra_quantiles

    def __call__(self, sample):
        return complex_normalisation(sample,self.bf_quantiles,self.bra_quantiles)

class SelectRandomTimepoint(object):
    """select a random timepoint form the time series

    time_dimension: int
        dimension index of time 
    """

    def __init__(self, time_dimension):
        self.td = time_dimension

    def __call__(self, sample):
        shape = sample.shape
        random_tp = np.random.randint(0,shape[self.td])
        slice_objects = [slice(0,shape[i]) if i != self.td else random_tp for i in range(len(shape))]
        return sample[slice_objects]

class CustomToTensor(object):
    """Normalise live TLS data with dimesnions t, c, y, x

    Args:
        bf_quantiles: list
            lower and upper quantiles for rescaling brightfield images (channel 0)
            Performed for each image individually
        bra_quantiles: list
            lower and upper quantiles for rescaling brachyury images (channel 1)
            rescaled across the timelapse
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        return from_numpy(sample)