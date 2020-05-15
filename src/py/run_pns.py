
import vtk
import numpy as np
import time
import LinearSubdivisionFilter as lsf

import argparse
import os

from scipy.optimize import least_squares
import math
import tensorflow as tf
import PNS
import itk

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Run PNS on encoded images that live on sphere', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', type=str, help='Directory with encoded images')
parser.add_argument('--out', type=str, help='Output csv', default="out.json")

args = parser.parse_args()

filenames = []

if(args.dir):
  replace_dir_name = args.dir
  normpath = os.path.normpath("/".join([args.dir, '**', '*']))
  for img in glob.iglob(normpath, recursive=True):
    if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
      fobj = {}
      fobj["img"] = img
      filenames.append(fobj)

def image_read(filename):

  print("Reading", filename);

  ImageType = itk.VectorImage[itk.F, 2]
  img_read = itk.ImageFileReader[ImageType].New(FileName=filename)
  img_read.Update()
  img = img_read.GetOutput()
  
  img_np = itk.GetArrayViewFromImage(img).astype(float)

  # Put the shape of the image in the json object if it does not exists. This is done for global information
  tf_img_shape = list(img_np.shape)
  if(tf_img_shape[0] == 1 and img.GetImageDimension() > 2):
    # If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
    tf_img_shape = tf_img_shape[1:]

  # This is the number of channels, if the number of components is 1, it is not included in the image shape
  # If it has more than one component, it is included in the shape, that's why we have to add the 1
  if(img.GetNumberOfComponentsPerPixel() == 1):
    tf_img_shape = tf_img_shape + [1]

  tf_img_shape = [1] + tf_img_shape

  return img, img_np, tf_img_shape

encoded_images = []

for img_obj in filenames:
  img, img_np, tf_img_shape = image_read(img_obj["img"])
  encoded_images.append(img_np.reshape(-1))


pns = PNS.PNS()

circle_center_v1 = pns.GetCircleCenter(3)
angle_r1 = pns.GetAngleR1(3)
rot_mat = pns.GetRotationMatrix(3)
points, projected, residuals = pns.GetScores(3)

out_dict = {}

out_dict["circle_center_v1"] = circle_center_v1.numpy().tolist()
out_dict["angle_r1"] = angle_r1
out_dict["rot_mat"] = rot_mat.numpy().tolist()
out_dict["points"] = points.numpy().tolist()
out_dict["projected"] = projected.numpy().tolist()
out_dict["residuals"] = residuals.numpy().tolist()

with open(args.out, 'w') as f:
    json.dump(out_dict, f)