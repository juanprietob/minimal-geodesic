
import time
import argparse
import os
import math
import numpy as np
import itk
import glob
import json

from sklearn.svm import OneClassSVM
import pickle

parser = argparse.ArgumentParser(description='Run PNS on encoded images that live on sphere', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', type=str, help='Directory with encoded images', required=True)
parser.add_argument('--split', type=float, help='Split the dataset [0-1], it will create a txt file with the file names of the split (0 = no split)', default=0)
parser.add_argument('--out', type=str, help='Output directory', default="./")

args = parser.parse_args()

out_dir = os.path.normpath(args.out)
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

filenames = []

if(args.dir):
  replace_dir_name = args.dir
  normpath = os.path.normpath("/".join([args.dir, '**', '*']))
  for img in glob.iglob(normpath, recursive=True):
    if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
      fobj = {}
      fobj["img"] = img
      filenames.append(fobj)


np.random.shuffle(np.array(filenames))

filenames_split = []

if args.split > 0:
  
  split_index = int(np.shape(filenames)[0]*args.split)
  filenames_eval = filenames[0:split_index]
  filenames = filenames[split_index:-1]

  out_train = os.path.join(out_dir, os.path.normpath(args.dir) + "_split_train.json") 
  out_eval = os.path.join(out_dir, os.path.normpath(args.dir) + "_split_eval.json")

  print("Writing:", out_train)
  with open(out_train, 'w') as out_train_f:
    json.dump(filenames, out_train_f)

  print("Writing:", out_eval)
  with open(out_eval, 'w') as out_eval_f:
    json.dump(filenames_eval, out_eval_f)


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

clf = OneClassSVM(gamma='scale', tol=0.000001).fit(encoded_images)

print("Fitting one class svm ...")
clf.fit(encoded_images)

one_class_out = os.path.join(out_dir, "one_class_svm.pkl")
print("Saving", one_class_out)
with open(one_class_out, 'wb') as out_pkl:
  pickle.dump(clf, out_pkl)