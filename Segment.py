import argparse
import itertools
from os import listdir
from os.path import isfile,join

import chainer
import numpy as np
import nibabel as nib
import pandas as pd

from utils import crop_patch, load_nifti


def main():


    trainingData_path = '/Users/Elaine/desktop/MICCAI/Training'
    n_tiles = [10,10,10]
    for imagename in listdir(trainingData_path):
        image, affine = load_nifti(join(trainingData_path,imagename), with_affine=True)
        centers = [[], [], []]
        for img_len, len_out, center, n_tile in zip(image.shape, output_shape, centers, n_tiles):
            assert img_len < len_out * n_tile, "{} must be smaller than {} x {}".format(img_len, len_out, n_tile)
            stride = int((img_len - len_out) / (n_tile - 1))
            center.append(len_out / 2)
            for i in range(n_tile - 2):
                center.append(center[-1] + stride)
            center.append(img_len - len_out / 2)
        output = np.zeros((dataset["n_classes"],) + image.shape[:-1])
        for x, y, z in itertools.product(*centers):
            patch = crop_patch(image, [x, y, z], input_shape)
            patch = np.expand_dims(patch, 0)
            patch = np.asarray(patch)
            slices_out = [slice(center - len_out / 2, center + len_out / 2) for len_out, center in zip(args.output_shape, [x, y, z])]
            slices_in = [slice((len_in - len_out) / 2, len_in - (len_in - len_out) / 2) for len_out, len_in, in zip(args.output_shape, args.input_shape)]
            output[slice(None), slices_out[0], slices_out[1], slices_out[2]] += chainer.cuda.to_cpu(
                vrn(patch).data[0, slice(None), slices_in[0], slices_in[1], slices_in[2]])
        y = np.argmax(output, axis=0)
        nib.save(
            nib.Nifti1Image(np.int32(y), affine),
            os.path.join(
                os.path.dirname(image_path),
                subject + args.output_suffix))


if __name__ == '__main__':
    main()
