import numpy as np
from skimage.exposure import rescale_intensity
from torch import from_numpy
from skimage.measure import centroid
import torch
import numbers
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torchvision.transforms import functional as F

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
        
        slice_objects = [
            random_tp if i == self.td else slice(0,shape[i]) for i in range(len(shape))
        ]
        return sample[slice_objects]

class SelectRandomTPNumpy(object):
    """select a random timepoint form the time series

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

class CustomToTensor(object):
    """Custom ToTensor: works with any shape and does not normalisation
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        return from_numpy(sample)

class CustomCropCentroid(object):
    def __init__(self,intensity_channel, channel_dim,crop_size):
        self.intensity_channel = intensity_channel
        self.channel_dim = channel_dim
        self.crop_size = crop_size

    def __call__(self, sample):
        #shape = sample.shape
        intensity_image = np.take(sample,[self.intensity_channel],axis=self.channel_dim).squeeze(self.channel_dim)
        cent = centroid(intensity_image)[-2:]

        cropped = crop_around_centroid_2D(sample,cent,self.crop_size,self.crop_size)

        return cropped

def crop_around_centroid_2D(image, centroid, crop_height = 800, crop_width = 800):
    half_wid = int(crop_width//2)
    half_hgt = int(crop_height//2)
    c_0, c_1 = [int(c) for c in centroid]
    
    if c_0-half_wid < 0:
        x_borders = np.amax(np.array([
            [c_0-half_wid,0],
            [c_0+half_wid,crop_width]]),axis = 1)
    else:
        x_borders = np.amin(np.array([
            [c_0-half_wid,image.shape[0]-crop_width],
            [c_0+half_wid,image.shape[0]]]),axis = 1)
    if c_1-half_wid < 0:
        y_borders = np.amax(np.array([
            [c_1-half_hgt,0],
            [c_1+half_hgt,crop_height]]),axis = 1)
    else:
        y_borders = np.amin(np.array([
            [c_1-half_hgt,image.shape[1]-crop_height],
            [c_1+half_hgt,image.shape[1]]]),axis = 1)

    cropped_img = np.take(image,np.arange(y_borders[0],y_borders[1],1),axis=-2)
    cropped_img = np.take(cropped_img,np.arange(x_borders[0],x_borders[1],1),axis=-1)
    return cropped_img

class CropAndReshapeTL(object):
    def __init__(self,intensity_channel, channel_dim,crop_size,time_dim):
        self.intensity_channel = intensity_channel
        self.channel_dim = channel_dim
        self.crop_size = crop_size


    def __call__(self, sample):
        #shape = sample.shape
        intensity_image = np.take(sample,[self.intensity_channel],axis=self.channel_dim).squeeze(self.channel_dim)
        cent = centroid(intensity_image)[-2:]

        cropped = crop_around_centroid_2D(sample,cent,self.crop_size,self.crop_size)

        return np.moveaxis(cropped,0,1)

class ColorJitterBrightfield(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
        channel_dim: int = 0,
    ) -> None:
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.channel_dim = channel_dim

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    @staticmethod
    def get_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        shape = img.shape
        
        others = img[1]
        outs =[]
        for tp in range(4):
            
            out = img[0,tp]
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    out = F.adjust_brightness(out, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    out = F.adjust_contrast(out, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    out = F.adjust_saturation(out, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    out = F.adjust_hue(out, hue_factor)
        outs = torch.concat(outs,dim=0)
        return torch.concat(
            [
                out.unsqueeze(self.channel_dim),
                others.unsqueeze(self.channel_dim)
            ],
            dim=self.channel_dim
        )


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s