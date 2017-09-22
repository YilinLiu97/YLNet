"" 
Copyright (c) 2016, Jose Dolz .All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.
Jose Dolz. Dec, 2016.
email: jose.dolz.upv@gmail.com
LIVIA Department, ETS, Montreal.
"""

import sys
import pdb
from os.path import isfile, join
import os
import numpy as np
import nibabel as nib
import scipy.io as sio


""" To print function usage """
def printUsage(error_type):
    if error_type == 1:
        print(" ** ERROR!!: Few parameters used.")
    else:
        print(" ** ERROR!!: ...") # TODO
        
    print(" ******** USAGE ******** ")
    print(" --- argv 1: Folder containing mr images")
    print(" --- argv 2: Folder to save corrected label images")
 

def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames
    
def checkAnotatedLabels(argv):
    # Number of input arguments
    #    1: Folder containing label images
    #    2: Folder to save corrected label images
    
    if len(argv) < 2:
        printUsage(1)
        sys.exit()
    
    imagesFolder = argv[0]
    imagesFolderdst  = argv[1]
    
    imageNames = getImageImageList(imagesFolder)
    printFileNames = False
    nib.Nifti1Header.quaternion_threshold = -6.401211e-06 #threshold
    for i_d in xrange(len(imageNames)) :
        imageFileName = imagesFolder + '/' + imageNames[i_d]
            
        [imageData,img_proxy] = nib.load(imageFileName, printFileNames)

        # Find voxels different to 0
        # assume voxels equal to 0 are outside my ROI 
        idx = np.where(imageData > 0)

        # Create ROI and assign those indexes to 1
        roiImage = np.zeros(imageData.shape,dtype=np.int8)
        roiImage[idx] = 1
        
        print(" ... Saving roi...")
        nameToSave =  imagesFolderdst + '/ROI_' + imageNames[i_d]
        imageTypeToSave = np.dtype(np.int8)
       
        #Generate the nii file
        niiToSave = nib.Nifti1Image(roiImage, img_proxy.affine)
        niiToSave.set_data_dtype(np.dtype(np.int8))

        dim = len(roiImage.shape)
        zooms = list(img_proxy.header.get_zooms()[:dim])
                     
        if len(zooms) < dim :
            zooms = zooms + [1.0]*(dim-len(zooms))
    
        niiToSave.header.set_zooms(zooms)
        nib.save(niiToSave, imageName)
    
        print "... Image succesfully saved..."
 

            
    print " ******************************************  PROCESSING LABELS DONE  ******************************************"
  
   
if __name__ == '__main__':
   checkAnotatedLabels(sys.argv[1:])
