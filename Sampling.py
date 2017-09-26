import numpy as np
import random
import math
import nibabel as nib

from scipy.ndimage import generic_filter
from scipy.stats import entropy

def findCenters(img, patch_size, num_patches):
   centralVoxelsIndexes_x = np.random.choice(img.get_shape()[0]-patch_size[0]/2, num_patches, replace=True)
   centralVoxelsIndexes_y = np.random.choice(img.get_shape()[1]-patch_size[1]/2, num_patches, replace=True)
   centralVoxelsIndexes_z = np.random.choice(img.get_shape()[2]-patch_size[2]/2, num_patches, replace=True)
   
   centers = np.transpose([centralVoxelsIndexes_x,centralVoxelsIndexes_y,centralVoxelsIndexes_z])
   return centers
        

def crop_patch(img, centers, patch_size, num_patches):
    """
    crop 3D patches from an nii file

    """
    imgdata = img.get_data()
    patches = []
    patch_size = np.reshape(np.repeat(patch_size, num_patches),[num_patches,len(patch_size)]) #reshape in order to have the same size as 'centers'
    count = 0
    for c, len_ in zip(centers, patch_size):
        
      r_x = len_[0]/2
      r_y = len_[1]/2
      r_z = len_[2]/2

      minx = c[0] - r_x #compute coordinates
      maxx = c[0] + r_x

      miny = c[1] - r_y
      maxy = c[1] + r_y

      minz = c[2] - r_z
      maxz = c[2] + r_z

      rmd = len_[0]%2 #Get the remainder, if there is any
      if all(v>=0 for v in [minx,miny,minz]) and all(v<=border for v, border in zip([maxx,maxy,maxz],imgdata.shape)):
         patch = imgdata[minx:maxx+rmd, miny:maxy+rmd, minz:maxz+rmd]
 
      else:
         minx, miny, minz = np.clip([minx,miny,minz],0,img.get_shape())
         maxx, maxy, maxz = np.clip([maxx,maxy,maxz],0, img.get_shape())
         
         patch = imgdata[minx:maxx+rmd, miny:maxy+rmd, minz:maxz+rmd]

      patches.append(patch)
 
    return np.array(patches)

def _entropy(values):
    probabilities = np.bincount(values.astype(np.int)) / float(len(values))
    return entropy(probabilities)

def local_entropy(img, kernel_radius=2):
    """
    Compute the local entropy for each pixel in an image or image stack using the neighbourhood specified by the kernel.

    Arguments:
    ----------
    img           -- 2D or 3D uint8 array with dimensions MxN or TxMxN, respectively.
                     Input image.
    kernel_radius -- int
                     Neighbourhood over which to compute the local entropy.

    Returns:
    --------
    h -- 2D or 3D uint8 array with dimensions MxN or TxMxN, respectively.
         Local entropy.

    """
    return generic_filter(img.astype(np.float), _entropy, size=2*kernel_radius)

def patches_by_entropy(patches):
    '''
    Finds high-entropy patches based on label, allows net to learn borders more effectively
    '''
    return local_entropy(patches)
    
def dice_coefficients(label1, label2, labels=None):
    if labels is None:
        labels = np.unique(np.hstack((np.unique(label1), np.unique(label2))))
    dice_coefs = []
    for label in labels:
        match1 = (label1 == label)
        match2 = (label2 == label)
        denominator = 0.5 * (np.sum(match1.astype(np.float)) + np.sum(match2.astype(np.float)))
        numerator = np.sum(np.logical_and(match1, match2).astype(np.float))
        if denominator == 0:
            dice_coefs.append(0.)
        else:
            dice_coefs.append(numerator / denominator)
    return dice_coefs

nib.Nifti1Header.quaternion_threshold = -6.401211e-06
vol = nib.load('/Users/Elaine/desktop/MICCAI/Training/1001_3.nii')
num_patches = 2000
patch_size = [27,27,27]
centers = findCenters(vol,patch_size,num_patches)

patches = crop_patch(vol,centers,patch_size,num_patches)
patches_highEntropy = [patches_by_entropy(patch) for patch in patches]

