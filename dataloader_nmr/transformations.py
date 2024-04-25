import numpy as np
from skimage.transform import resize
#from sklearn.externals._pilutil import bytescale
import albumentations as A
import random
import cv2

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    deprecated from sklearn.externals._pilutil import bytescale
    #https://gemfury.com/aaronreidsmith/python:scikit-learn/scikit_learn-0.22-pp371-pypy3_71-linux_x86_64.whl/content/externals/_pilutil.py      

    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.

    This function is only available if Python Imaging Library (PIL) is installed.

    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.

    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)

    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def create_dense_target(tar: np.ndarray): # flaw here - if the target doesn't contain all classess, their numeration scramble
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy

def normalize_01(inp: np.ndarray):
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out

def normalize_01_simple(inp: np.ndarray):
    inp_out = inp / 255
    return inp_out

def normalize(inp: np.ndarray, mean: float, std: float):
    inp_out = (inp - mean) / std
    return inp_out


def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    """Normalize the data to a certain range. Default: [0-255]"""   
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self): return str([transform for transform in self.transforms])


class MoveAxis:
    """From [H, W, C] to [C, H, W]"""

    def __init__(self, transform_input: bool = True, transform_target: bool = False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        if self.transform_input: inp = np.moveaxis(inp, -1, 0)
        if self.transform_target: tar = np.moveaxis(tar, -1, 0)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class DenseTarget:
    """Creates segmentation maps with consecutive integers, starting from 0"""

    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        tar = create_dense_target(tar)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Resize:
    """Resizes the image and target - based on skimage"""

    def __init__(self,
                 input_size: tuple,
                 target_size: tuple,
                 input_kwargs: dict = {},
                 target_kwargs: dict = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
                 ):
        self.input_size = input_size
        self.target_size = target_size
        self.input_kwargs = input_kwargs
        self.target_kwargs = target_kwargs

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        self.input_dtype = inp.dtype
        self.target_dtype = tar.dtype

        inp_out = resize(image=inp,
                         output_shape=self.input_size,
                         **self.input_kwargs
                         )
        tar_out = resize(image=tar,
                         output_shape=self.target_size,
                         **self.target_kwargs
                         ).astype(self.target_dtype)
        return inp_out, tar_out

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Normalize01:
    """Squash image input to the value range [0, 1] (no clipping)"""

    def __init__(self):
        pass

    def __call__(self, inp, tar):
        inp = normalize_01(inp)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class Normalize01_simple:
    """Squash image input to the value range [0, 1] (no clipping)"""

    def __init__(self):
        pass

    def __call__(self, inp, tar):
        inp = normalize_01_simple(inp)
        tar = normalize_01_simple(tar)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Normalize:
    """Normalize based on mean and standard deviation."""

    def __init__(self,
                 mean: float,
                 std: float
                 ):
        self.mean = mean
        self.std = std

    def __call__(self, inp, tar):
        inp = normalize(inp, self.mean, self.std)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class AlbuSeg2d:
    def __init__(self, albu):
        self.albu = albu

    def __call__(self, inp, tar):
        # input, target
        out_dict = self.albu(image=inp, mask=tar)
        input_out = out_dict['image']
        target_out = out_dict['mask']

        return input_out, target_out

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class RandomCrop:
    """Wrapper around Albumentation RandomCrop"""

    def __init__(self, width=512, height=512, transform_input: bool = True, transform_target: bool = True):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.cropper = A.RandomCrop(width=width, height=height)

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        cropped_dict = self.cropper(image=inp, mask=tar)
        if self.transform_input: inp = cropped_dict['image']
        if self.transform_target: tar = cropped_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class CenterCrop:
    """Wrapper around Albumentation CenterCrop"""

    def __init__(self, width=512, height=512, transform_input: bool = True, transform_target: bool = True):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.cropper = A.CenterCrop(width=width, height=height)

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        cropped_dict = self.cropper(image=inp, mask=tar)
        if self.transform_input: inp = cropped_dict['image']
        if self.transform_target: tar = cropped_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class HorizontalFlip:
    """Wrapper around Albumentation HorizontalFlip"""

    def __init__(self, width=512, height=512, transform_input: bool = True, transform_target: bool = True, p = 0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.flipper = A.HorizontalFlip(p = p)

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        flipped_dict = self.flipper(image=inp, mask=tar)
        if self.transform_input: inp = flipped_dict['image']
        if self.transform_target: tar = flipped_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class RandomScale:
    """Wrapper around Albumentation HorizontalFlip"""

    def __init__(self, width=512, height=512, transform_input: bool = True, 
                 transform_target: bool = True, 
                 scale_limit = [0.7, 1.],
                 p = 0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.former = A.RandomScale(scale_limit = scale_limit, 
                                       interpolation=0, #cv2.INTER_NEAREST
                                       p = p)
       

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        form_dict = self.former(image=inp, mask=tar)
        if self.transform_input: inp = form_dict['image']
        if self.transform_target: tar = form_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class RandomBrightnessContrast:
    """Wrapper around RandomBrightnessContrast"""

    def __init__(self, transform_input: bool = True, 
                 transform_target: bool = False, 
                 brightness_limit=0.2, 
                 contrast_limit=0.1, 
                 brightness_by_max=True,
                 p = 0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.former = A.RandomBrightnessContrast(brightness_limit = brightness_limit, 
                                       contrast_limit=contrast_limit,
                                       brightness_by_max = brightness_by_max, 
                                       p = p)
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        form_dict = self.former(image=inp, mask=tar)
        if self.transform_input: inp = form_dict['image']
        if self.transform_target: tar = form_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__}) 

class Flip:
    """Wrapper around Albumentation Flip"""

    def __init__(self, transform_input: bool = True, transform_target: bool = True, p = 0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.flipper = A.Flip(p = p)

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        flipped_dict = self.flipper(image=inp, mask=tar)
        if self.transform_input: inp = flipped_dict['image']
        if self.transform_target: tar = flipped_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__}) 

class RandomRotate90:
    """Wrapper around Albumentation RandomScale"""

    def __init__(self, transform_input: bool = True, 
                 transform_target: bool = True,
                 p = 0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.former = A.RandomRotate90(p = p)
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        form_dict = self.former(image=inp, mask=tar)
        if self.transform_input: inp = form_dict['image']
        if self.transform_target: tar = form_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__}) 

class ShiftScaleRotate:
    """Wrapper around Albumentation ShiftScaleRotate"""

    def __init__(self, transform_input: bool = True, 
                 transform_target: bool = True,
                 shift_limit=0.0625, 
                 scale_limit=0.1, 
                 rotate_limit=45, 
                 interpolation=0, # INTER_NEAREST
                 border_mode=0, # BORDER_CONSTANT 
                 value=0,    #padding value
                 mask_value=None, 
                 p=0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.former = A.ShiftScaleRotate (shift_limit=shift_limit, scale_limit=scale_limit, 
                                          rotate_limit=rotate_limit, interpolation=interpolation, 
                                          border_mode=border_mode, value=value, p=0.5)
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        form_dict = self.former(image=inp, mask=tar)
        if self.transform_input: inp = form_dict['image']
        if self.transform_target: tar = form_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__}) 

class ShiftScaleRotateCustom:
    """Wrapper around Albumentation ShiftScaleRotate. Works only on source images"""

    def __init__(self, transform_input: bool = True, 
                 transform_target: bool = True,
                 shift_limit=0.0625, 
                 scale_limit=0.1, 
                 rotate_limit=0, 
                 interpolation=0, # INTER_NEAREST
                 border_mode=0, # BORDER_CONSTANT 
                 value=0,    #padding value
                 mask_value=None, 
                 p=0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.former = A.ShiftScaleRotate (shift_limit=shift_limit, scale_limit=scale_limit, 
                                          rotate_limit=rotate_limit, interpolation=interpolation, 
                                          border_mode=border_mode, value=value, p=0.5)
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        #form_dict = self.former(image=inp, mask=inp)
        form_dict = self.former(image=inp, mask=inp)
        if self.transform_input: inp = form_dict['image']
        if self.transform_target: tar = form_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class AddGaussianNoise():
    # part of code copied from 
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py
    def __init__(self, 
                 mean=0., 
                 var_limit=1.,
                 clip_min = 0.0,
                 clip_max = None,
                 p = 0.5
                 ):
        
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )        
    
        self.mean = mean
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.p = p
        
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        prob = random.uniform(0, 1)
        if prob < self.p:
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            std = var ** 0.5
            mean = random.uniform(0, self.mean)
            noise = np.random.randn(*inp.shape) * std + mean
            noise = np.clip(a = noise, a_min = self.clip_min, a_max = self.clip_max)
            inp = inp + noise
        
        return inp, tar
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, var_limit={1})'.format(self.mean, self.var_limit)

class GaussianBlur:
    """Wrapper around Albumentation GaussianBlur"""

    def __init__(self, transform_input: bool = True, 
                 transform_target: bool = False,
                 blur_limit=(1, 5), 
                 sigma_limit=3,                 
                 p=0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target        
        self.former = A.GaussianBlur( blur_limit=blur_limit, sigma_limit=sigma_limit, p=p)
    def __call__(self, inp: np.ndarray, tar: np.ndarray): 
        form_dict = self.former(image=inp, mask=tar)
        if self.transform_input: inp = form_dict['image']
        if self.transform_target: tar = form_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class AddRandomBrightness():

    def __init__(self, 
                 amount_range=0.2, # as percentage from max value in range
                 p = 0.5
                 ):   
        self.amount_range = amount_range
        self.p = p
        
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        prob = random.uniform(0, 1)
        if prob < self.p:
            rnge = np.ptp(inp)
            amount = random.uniform(0, self.amount_range) * rnge
            inp = inp + amount            
        return inp, tar
    
    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class AddSpuriousNoise():
    def __init__(self, 
                 mean=10., 
                 var_limit=4.,
                 clip_min = -1,
                 clip_max = 15,
                 frame = 34,
                 p = 0.5
                 ):
        
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )        
    
        self.mean = mean
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.p = p
        self.frame = frame
        
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        prob = random.uniform(0, 1)
        #pp(prob)
        if prob < self.p:
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            std = var ** 0.5            
            mean = random.uniform(0, self.mean)            
            
            noise = np.random.randn(inp.shape[0], inp.shape[1]) * std + mean
            noise = np.clip(a = noise, a_min = self.clip_min, a_max = self.clip_max)
            noise = cv2.GaussianBlur(noise,(3,3),0)
            #framing a black (zero) border to mimic the real samples which underwent framing as well
            noise[:self.frame, :] = 0
            noise[:, :self.frame] = 0
            noise[inp.shape[0]-self.frame :, :] = 0
            noise[:, inp.shape[1]-self.frame:] = 0
            
            #pp("shape, min, max, mean, std")
            #pp(noise.shape, noise.min(), noise.max(), noise.mean(), noise.std())
            
            num_of_frames = np.random.choice(8, 1, p=[0.0, 0.0, 0.0, 0.0, 0.65, 0.2, 0.1, 0.05])[0]
            
            # coefficient to decay the noise from the central frame
            ran = np.array(list(range(num_of_frames)))
            ran =  np.square(ran - ran.mean())
            ran = (ran.max() - ran) +1
            ran = ran/ran.max()   # to make it look like [0.2, 0.8, 1. , 0.8, 0.2]
            
            # starting point to introduce noise in the stack of frames
            # min/max starting somewhere in the middleof frame stack
            middle = inp.shape[2]//2
            start = np.random.randint(middle-num_of_frames, middle, 1)[0]            
            '''
            pp("var, std, mean")
            pp(var, std)
            pp(mean)
            pp("num_of_frames", num_of_frames)
            pp("ran", ran)
            pp("start", start)
            '''
            for i, dx in enumerate(range(start, start+num_of_frames)):
                if (dx >= 0 and dx < inp.shape[2]):
                    ns = noise*ran[i]
                    inp[:,:,dx] = inp[:,:,dx]* (1-ran[i])
                    inp[:,:,dx] = inp[:,:,dx]+ns            
        
        return inp, tar
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, var_limit={1})'.format(self.mean, self.var_limit)
        
class AddSpuriousNoise_archived():
    def __init__(self, 
                 mean=10., 
                 var_limit=4.,
                 clip_min = -1,
                 clip_max = 15,
                 frame = 34,
                 p = 0.5
                 ):
        
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )        
    
        self.mean = mean
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.p = p
        self.frame = frame
        
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        prob = random.uniform(0, 1)
        #pp(prob)
        if prob < self.p:
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            std = var ** 0.5            
            mean = random.uniform(0, self.mean)            
            
            noise = np.random.randn(inp.shape[0], inp.shape[1]) * std + mean
            noise = np.clip(a = noise, a_min = self.clip_min, a_max = self.clip_max)
            noise = cv2.GaussianBlur(noise,(3,3),0)
            #framing a black (zero) border to mimic the real samples which underwent framing as well
            noise[:self.frame, :] = 0
            noise[:, :self.frame] = 0
            noise[inp.shape[0]-self.frame :, :] = 0
            noise[:, inp.shape[1]-self.frame:] = 0
            
            #pp("shape, min, max, mean, std")
            #pp(noise.shape, noise.min(), noise.max(), noise.mean(), noise.std())
            
            num_of_frames = np.random.choice(8, 1, p=[0.0, 0.0, 0.0, 0.0, 0.65, 0.2, 0.1, 0.05])[0]
            
            # coefficient to decay the noise from the central frame
            ran = np.array(list(range(num_of_frames)))
            ran =  np.square(ran - ran.mean())
            ran = (ran.max() - ran) +1
            ran = ran/ran.max()   # to make it look like [0.2, 0.8, 1. , 0.8, 0.2]
            
            # starting point to introduce noise in the stack of frames
            # min start from 0-num_of_frames 
            start = np.random.randint(0-num_of_frames, 20, 1)[0]            
            '''
            pp("var, std, mean")
            pp(var, std)
            pp(mean)
            pp("num_of_frames", num_of_frames)
            pp("ran", ran)
            pp("start", start)
            '''
            for i, dx in enumerate(range(start, start+num_of_frames)):
                if (dx >= 0 and dx < 21):
                    ns = noise*ran[i]
                    inp[:,:,dx] = inp[:,:,dx]* (1-ran[i])
                    inp[:,:,dx] = inp[:,:,dx]+ns            
        
        return inp, tar
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, var_limit={1})'.format(self.mean, self.var_limit)       