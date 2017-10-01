import numpy as np
import random
import math
import nibabel as nib
from os import listdir
from os.path import isfile,join
import torch

from scipy.ndimage import generic_filter
from scipy.stats import entropy

def findCenters(img, patch_size, num_patches):

   centralVoxelsIndexes_x = np.random.choice(img.shape[0]-(patch_size[0]/2), num_patches, replace=True)
   centralVoxelsIndexes_y = np.random.choice(img.shape[1]-(patch_size[1]/2), num_patches, replace=True)
   centralVoxelsIndexes_z = np.random.choice(img.shape[2]-(patch_size[2]/2), num_patches, replace=True)
   
   centers = np.transpose([centralVoxelsIndexes_x,centralVoxelsIndexes_y,centralVoxelsIndexes_z])
   return centers
        

def crop_patch(img, centers, patch_size):
    """
    crop 3D patches from an nii file
    """
    imgdata = img.get_data()
    patches = []
    num_patches = centers.shape[0]
    patch_size = np.reshape(np.repeat(patch_size, num_patches),[num_patches,len(patch_size)]) #reshape in order to have the same size as 'centers'
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
      if all(v>=0 for v in [minx,miny,minz]) and all(v<border for v, border in zip([maxx,maxy,maxz],imgdata.shape)):
         patch = imgdata[minx:(maxx+rmd), miny:(maxy+rmd), minz:(maxz+rmd)]
 
      else:
         minx, miny, minz = np.clip([minx,miny,minz],0,img.get_shape())
         maxx, maxy, maxz = np.clip([maxx,maxy,maxz],0, img.get_shape())
         
         patch = imgdata[minx:maxx+rmd, miny:maxy+rmd, minz:maxz+rmd]
      if np.percentile(patch,10) != 0:#exclude patches in which more than 25% of the voxels were of zero-intensity
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

def patches_by_entropy(img, num_patches, patches, patch_size):
    '''
    Finds high-entropy patches based on label, allows net to learn borders more effectively
    '''
    imgdata = img.get_data()
    patches_highEn = []
    i = 0
    for patch in patches:
       l_ent = local_entropy(patch)
       top_ent = np.percentile(l_ent, 80)


       if top_ent != 0:#otherwise, resample

          highest = np.argwhere(l_ent >= top_ent)

          p_r = random.sample(highest,3)

          for c, len_ in zip(np.array(p_r), [20,20,20]):
        
               minx = c[0] - len_/2 #compute coordinates
               maxx = c[0] + len_/2

               miny = c[1] - len_/2
               maxy = c[1] + len_/2

               minz = c[2] - len_/2
               maxz = c[2] + len_/2

               if all(v>=0 for v in [minx,miny,minz]) and all(v<=border for v, border in zip([maxx,maxy,maxz],imgdata.shape)):
                  patch = imgdata[minx:maxx, miny:maxy, minz:maxz]

 #                 print 'patch_highEn ', np.array(patch)
 
               else:
                  minx, miny, minz = np.clip([minx,miny,minz],0,img.get_shape())
                  maxx, maxy, maxz = np.clip([maxx,maxy,maxz],0, img.get_shape())
                  patch = imgdata[minx:maxx, miny:maxy, minz:maxz]
 #                 print 'patch_highEn ', np.array(patch)
               patches_highEn.append(patch)
    return np.array(patches_highEn)
    
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

def make_training_samples(imgname,labelname,num_patches,patch_size):
   imgPatches = []
   labelPatches = []
   nib.Nifti1Header.quaternion_threshold = -6.401211e-06
   vol = nib.load(imgname)
   label = nib.load(labelname)
   affine = vol.affine

   centers = findCenters(vol,patch_size,num_patches)
   img_patches = crop_patch(vol,centers,patch_size)
   label_patches = crop_patch(label,centers,patch_size)

      
   return np.array(img_patches), np.array(label_patches)
   


