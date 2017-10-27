import numpy as np
import random
import math
import nibabel as nib
import os
from os import listdir
from os.path import isfile,join
import torch

from scipy.ndimage import generic_filter
from scipy.stats import entropy

# ************************** Pre-processing *******************************
def normalization(img):
   img_normed = []
   imgdata = img.get_data()
   mean = np.mean(imgdata)
   std = np.std(imgdata)
   
   img_normed = (imgdata-mean)/std
   return np.array(img_normed)

#img = nib.load('/Users/Elaine/desktop/Dataset/Training/subject205_noskl_mid_s205abcd_superseg_contrasted_path.nii')
#normed = normalization(img)  
# ************************** For Training *********************************
def findROICenters(labelData,num_classes,num_patches):
   ROICenters = []
   #n_perClass = int(num_patches/(num_classes-1))
   for p_ix in xrange(1,num_classes):
      centers = np.argwhere(p_ix==labelData)
      centers_toChoose = np.random.permutation(centers)
      ROICenters.append(centers_toChoose[0:num_patches])
   return np.array(ROICenters)

def findCenters(img, patch_size,num_patches):

   centralVoxelsIndexes_x = np.random.choice(img.shape[0]-(patch_size[0]/2), num_patches, replace=True)
   centralVoxelsIndexes_y = np.random.choice(img.shape[1]-(patch_size[1]/2), num_patches, replace=True)
   centralVoxelsIndexes_z = np.random.choice(img.shape[2]-(patch_size[2]/2), num_patches, replace=True)
   
   centers = np.transpose([centralVoxelsIndexes_x,centralVoxelsIndexes_y,centralVoxelsIndexes_z])

   return np.array(centers)

def crop_patch(imgdata, centers, patch_size):
    """
    crop 3D patches from an nii file
    """
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
      if all(v>=0 for v in [minx,miny,minz]) and all(v<=border-1 for v, border in zip([maxx,maxy,maxz],imgdata.shape)):
         patch = imgdata[minx:(maxx+rmd), miny:(maxy+rmd), minz:(maxz+rmd)]
         patches.append(patch)
      '''
      else:
         
         minx, miny, minz = np.clip([minx,miny,minz],0,img.get_shape())
         maxx, maxy, maxz = np.clip([maxx,maxy,maxz],0, img.get_shape())
         patch = imgdata[minx:maxx+rmd, miny:maxy+rmd, minz:maxz+rmd]
         '''
 #     if np.percentile(patch,10) != 0:#exclude patches in which more than 25% of the voxels were of zero-intensity
 
    return patches

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

def patches_by_entropy(imgdata, num_patches, patches, patch_size):
    '''
    Finds high-entropy patches based on label, allows net to learn borders more effectively
    '''
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

def make_training_samples(imgname,labelname,num_patches,patch_size,num_classes):
   imgPatches = []
   labelPatches = []

   vol = nib.load(imgname)
   label = nib.load(labelname)
   affine = vol.affine

   vol = normalization(vol)
#   vol = vol.get_data() 
   label = np.array(label.get_data())
#   num_patches_rand = int(num_patches/2)
#   num_patches_roi = int(num_patches-num_patches_rand)
   num_p = int(num_patches/num_classes)
   
   rand_centers = findCenters(vol,patch_size,num_p)
   roi_centers = findROICenters(label,num_classes,num_p)
   
   roi_img_patches,roi_label_patches = [],[]
      
   vol_patches_1 = crop_patch(vol,roi_centers[0],patch_size)
   label_patches_1 = crop_patch(label,roi_centers[0],patch_size)
   vol_patches_2 = crop_patch(vol,roi_centers[1],patch_size)
   label_patches_2 = crop_patch(label,roi_centers[1],patch_size)
 #  print('vol_patches_1.shape ',len(vol_patches_1))
#   print('vol_patches_2.shape ',len(vol_patches_2))
   diff = abs(len(vol_patches_1)-len(vol_patches_2))
   if len(vol_patches_1)<len(vol_patches_2):
      label_patches_2 = label_patches_2[0:len(label_patches_2)-diff]
      vol_patches_2 = vol_patches_2[0:len(vol_patches_2)-diff]
   else:
      label_patches_1 = label_patches_1[0:len(label_patches_1)-diff]
      vol_patches_1 = vol_patches_1[0:len(vol_patches_1)-diff]
   roi_img_patches.append(vol_patches_1)
   roi_img_patches.append(vol_patches_2)
   roi_label_patches.append(label_patches_1)
   roi_label_patches.append(label_patches_2)
      
   roi_img_patches = np.reshape(roi_img_patches,[-1,patch_size[0],patch_size[1],patch_size[2]])
   roi_label_patches = np.reshape(roi_label_patches,[-1,patch_size[0],patch_size[1],patch_size[2]])

   rand_img_patches = crop_patch(vol,rand_centers,patch_size)
   rand_label_patches = crop_patch(label,rand_centers,patch_size)

   img_patches = rand_img_patches
   label_patches = rand_label_patches
   img_patches.extend(roi_img_patches)
   label_patches.extend(roi_label_patches)


   return affine, np.random.permutation(np.array(img_patches)), np.random.permutation(np.array(label_patches))
'''
vol = nib.load('/Users/Elaine/desktop/Dataset/Training/subject205_noskl_mid_s205abcd_superseg_contrasted_path.nii')
patch_size = [27,27,27]
num_classes = 11
num_patches = 1000
label = nib.load('/Users/Elaine/desktop/Dataset/Labels/subject205_RIGHT_all_labels_8bit_path_RightLeftAmygdalaSubfields.nii')
ROICenters= findROICenters(label,num_classes,num_patches)
roi_centers = findROICenters(label,num_classes,num_patches)
   
roi_img_patches,roi_label_patches = [],[]
for i in xrange(0,len(roi_centers)):
   roi_img_patches.append(crop_patch(vol,roi_centers[i],patch_size))
   roi_label_patches.append(crop_patch(label,roi_centers[i],patch_size))
      
roi_img_patches = np.reshape(roi_img_patches,[-1,patch_size[0],patch_size[1],patch_size[2]])
roi_label_patches = np.reshape(roi_label_patches,[-1,patch_size[0],patch_size[1],patch_size[2]])
#affine,img_patches,label_patches = make_training_samples('/Users/Elaine/desktop/Dataset/Training/subject205_noskl_mid_s205abcd_superseg_contrasted_path.nii','/Users/Elaine/desktop/Dataset/Labels/subject205_RIGHT_all_labels_8bit_path_RightLeftAmygdalaSubfields.nii',1000,[27,27,27],11)

nib.Nifti1Header.quaternion_threshold = -6.401211e-06
vol = nib.load('/Users/Elaine/desktop/MICCAI_old/Training/1002_3.nii')
affine = vol.affine
centers = findCenters(vol,[27,27,27],1000)
patches = crop_patch(vol,centers,[27,27,27])
nib.save(nib.Nifti1Image(np.int32(y), affine),'/Users/Elaine/desktop/patchVis.nii')
'''

#************************* For Testing **************************************
def crop_det_patches(img,patch_size):
    samplesCoords,patches = [],[]
    imgdata = img.get_data()
    imgDims = list(imgdata.shape)

    zMinNext=0
    zCentPredicted = False

    while not zCentPredicted:
       zMax = min(zMinNext+patch_size[2], imgDims[2])
       zMin = zMax-patch_size[2]
       zMinNext = zMax

       if zMax<imgDims[2]:
          zCentPredicted = False
       else:
          zCentPredicted = True

       yMinNext=0
       yCentPredicted = False

       while not yCentPredicted:
          yMax = min(yMinNext+patch_size[1],imgDims[1])
          yMin = yMax - patch_size[1]
          yMinNext = yMax

          if yMax < imgDims[1]:
             yCentPredicted=False
          else:
             yCentPredicted=True

          xMinNext = 0
          xCentPredicted = False

          while not xCentPredicted:
             xMax = min(xMinNext+patch_size[0],imgDims[0])
             xMin = xMax - patch_size[0]
             xMinNext = xMax

             if xMax < imgDims[0]:
                xCentPredicted = False
             else:
                xCentPredicted = True

             samplesCoords.append([[xMin,xMax],[yMin,yMax],[zMin,zMax]])
             patches.append(imgdata[xMin:xMax,yMin:yMax,zMin:zMax])

    return np.array(samplesCoords),np.array(patches)

    

def stitch_Patches(probMaps,samplesCoords_perBatch,predPatches_perBatch,batch_size):#per_Batch
   
   for i in xrange(batch_size):
      samplesCoords_i = samplesCoords_perBatch[i]
      min_coords = [samplesCoords_i[0][0],samplesCoords_i[1][0],samplesCoords_i[2][0]]
      max_coords = [(x+25) for x in min_coords]
   
#      print('predPatches_perBatch[i].shape ',np.array(predPatches_perBatch[i]).shape)
#      print('min_coords[0]:max_coords[0] ',np.array(probMaps[min_coords[0]:max_coords[0]]).shape)
      probMaps[min_coords[0]:max_coords[0],min_coords[1]:max_coords[1],min_coords[2]:max_coords[2]] = predPatches_perBatch[i]
      
   return np.array(probMaps)

def saveImage(probMaps,affine):
   # Generate folder for output files
   BASE_DIR = os.getcwd()
   out_dir = os.path.join(BASE_DIR,'ValiRes')
   if not os.path.exists(out_dir):
      os.mkdir(out_dir)
   pred_vol = nib.Nifti1Image(probMaps,affine)
   nib.save(pred_vol, os.path.join(out_dir,'segRes.nii'))

'''
label = nib.load('/home/yilin/Dataset/Labels/subject205_RIGHT_all_labels_8bit_path_RightLeftAmygdalaNOSUBFIELDS.nii')
affine = label.affine
coordinates,patches = crop_det_patches(label,[25,25,25])
probMaps = np.array(np.zeros(list(label.get_shape())), dtype="int16")
next = 0
batch_indices= 0
for i in xrange(0,len(patches),50):
   if i != 0:
      batch_indices = i
      probMaps = stitch_Patches(probMaps,coordinates[next:batch_indices],patches[next:batch_indices],50)
   next = batch_indices
saveImage(probMaps,affine) 
'''

  
def make_testing_samples(imgname,labelname,patch_size):
   imgPatches = []
   labelPatches = []
   
   #nib.Nifti1Header.quaternion_threshold = -6.401211e-06
   vol = nib.load(imgname)
   label = nib.load(labelname)
   affine = vol.affine

   img_patches = crop_det_patches(vol,centers,patch_size)
   label_patches = crop_det_patch(label,centers,patch_size)
      
   return affine, np.array(img_patches), np.array(label_patches)


def computeDice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())
   


