
import numpy as np
import time

import argparse
import os

from scipy.optimize import least_squares
import math
import tensorflow as tf
import PNS
import itk
import glob
import json

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='Run PNS on encoded images that live on sphere', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', type=str, help='Directory with encoded images', required=True)
parser.add_argument('--continue_fitting', type=bool, help='Continue the fitting process', default=False)
parser.add_argument('--out', type=str, help='Output directory', default="./out")

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


encoded_images = np.array(encoded_images).astype('float32')

print(encoded_images.shape)
points = encoded_images

pns = PNS.PNS()
pns.SetOutputDirectory(args.out)
pns.Fit(points, args.continue_fitting)
points = pns.GetFkPoints(3)
projected = pns.GetProjectedPoints(3)
circle_center_v1, angle_r1, rot_mat = pns.GetSubSphereFit(3)


out_dir = os.path.normpath(args.out)
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

out_obj = {}
out_obj["circle_center_v1"] = circle_center_v1.numpy().tolist()
out_obj["angle_r1"] = angle_r1.numpy().tolist()
out_obj["rot_mat"] = rot_mat.numpy().tolist()
out_obj["pns"] = []

for name, p, pr in zip(filenames, points, projected):
  o_obj = {}
  o_obj["name"] = name
  o_obj["point"] = p.numpy().tolist()
  o_obj["projected"] = pr.numpy().tolist()
  out_obj["pns"].append(o_obj)

out_pns = os.path.normpath(args.out) + ".json"
print("Writting", out_pns)
with open(out_pns, 'w') as f:
  json.dump(out_obj, f)