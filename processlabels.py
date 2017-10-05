import nibabel as nib
import numpy as np
import os
from os.path import join
from os import listdir

def load_nii(filename):
    #for MICCAI
    nib.Nifti1Header.quaternion_threshold = -6.401211e-06
    vol = nib.load(filename)
    affine = vol.affine
    return np.array(vol.get_data()),affine
    

def checkAnotatedLabels(label_path,labelsToSelect):
    
    imgNames = listdir(label_path)

    for i_d in xrange(0, len(imgNames)):
        
        imgdata,affine = load_nii(join(label_path,imgNames[i_d]))

        labelsOrig = np.unique(imgdata)


        #Correct labels
        labelCorrectedImage = np.zeros(imgdata.shape,dtype=np.int8)
        for i in xrange(0,len(labelsToSelect)):
            idx = np.where(imgdata == labelsToSelect[i])
            labelCorrectedImage[idx] = i+1


        print("...Saving labels...")
        niiToSave = nib.Nifti1Image(labelCorrectedImage, affine)
        nameToSave = label_path + '/' + imgNames[i_d]
        nib.save(niiToSave,nameToSave)
        

    print " ******************************************  PROCESSING LABELS DONE  ******************************************"
