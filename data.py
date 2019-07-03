from __future__ import print_function

import os
import numpy as np
import pdb
import cv2
from fnmatch import fnmatch
from skimage.io import imsave, imread
import pickle
import imageio
import matplotlib.pyplot as plt

#Prepare training and test set
def create_train_data(netparms):
    data_path=netparms.data_path
    filenames_img = []
    filenames_mask = []
    # train_data_path = os.path.join(data_path, 'train')
    if  os.path.exists('imgs_trainPath.npy')==True and os.path.exists('imgs_mask_trainPath.npy')==True :
        print('Training set already exists and loaded from file')
        return
    Gpaths=[x for x in next(os.walk(data_path))][1]
    Gpaths=[os.path.join(data_path,x) for x in Gpaths]

    #pdb.set_trace()
    images = os.listdir(data_path)
    total =sum(len(os.listdir(os.path.join(y,'ground_truth'))) for y in (Gpaths))
    i = 0
    print('-'*30)
    print('Creating trainig images...')
    print('-'*30)
    img_mask=[]
    for video_number in range(len(images)):
        for image_gt_name in os.listdir(os.path.join(Gpaths[video_number],'ground_truth')):
            #pdb.set_trace()
            name_gt=os.path.join(Gpaths[video_number],'ground_truth', image_gt_name)
            name_left=os.path.join(Gpaths[video_number],'left_frames', os.path.splitext(image_gt_name)[0]+'.jpg')
            name_right=os.path.join(Gpaths[video_number],'right_frames', os.path.splitext(image_gt_name)[0]+'.jpg')
            img_gt = imread(name_gt)
            img_left= imread(name_left)
            img_right= imread(name_right)
            #pdb.set_trace()
            try:
                filenames_img.append((name_left, name_right))
              #  pdb.set_trace()
                filenames_mask.append(name_gt)
            except ValueError:
                pdb.set_trace()

            if i % 1000 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
            if i == total:
                print('Loading done.')
                np.save('imgs_trainPath.npy', filenames_img)
                np.save('imgs_mask_trainPath.npy', filenames_mask)
                print('Saving to .npy files done.')
                print('Loading done.')
                return

def load_train_data():
    imgs_train = np.load('imgs_trainPath.npy')
    imgs_mask_train = np.load('imgs_mask_trainPath.npy')
    return imgs_train, imgs_mask_train


def create_test_data(netparams):
    filenames_img = []
    filenames_mask = []
    data_path_test=netparams.data_path_test
    if  os.path.exists('imgs_test.npy')==True and os.path.exists('imgs_id_test.npy')==True :
        print('Test set already exists and loaded from file')
        return 
    Gpaths=[x for x in next(os.walk(data_path_test))][1]
    Gpaths=[os.path.join(data_path_test,x) for x in Gpaths]

    images = os.listdir(data_path_test)
    total =sum(len(os.listdir(os.path.join(y,'left_frames'))) for y in (Gpaths))
    i = 0
    print('-'*30)
    print('Creating trainig images...')
    print('-'*30)
    img_mask=[]
    for video_number in range(len(images)):
        for image_gt_name in os.listdir(os.path.join(Gpaths[video_number],'ground_truth')):
            #pdb.set_trace()
            name_gt=os.path.join(Gpaths[video_number],'ground_truth', image_gt_name)
            name_left=os.path.join(Gpaths[video_number],'left_frames', os.path.splitext(image_gt_name)[0]+'.jpg')
            name_right=os.path.join(Gpaths[video_number],'right_frames', os.path.splitext(image_gt_name)[0]+'.jpg')
            img_gt = imread(name_gt)
            img_left= imread(name_left)
            img_right= imread(name_right)
            #pdb.set_trace()           
            try:
                filenames_img.append((name_left, name_right))
                filenames_mask.append(name_gt)
            except ValueError:
                pdb.set_trace()

            if i % 1000 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
            if i == total:
                print('Loading done.')
                np.save('imgs_test.npy', filenames_img)
                np.save('imgs_id_test.npy', filenames_mask)
                print('Saving to .npy files done.')
                print('Loading done.')
                return


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id
def plot_imagesT(images,images_right, cls_true, cls_pred=None, smooth=True, filename='test.png'):
    #pdb.set_trace()
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(4, 6,figsize=(60, 60))
   
    if cls_pred is None:
        hspace = 0.2
    else:
        hspace = 0.2
    fig.subplots_adjust(hspace=hspace, wspace=0.1)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    count1 =0
    count2 =0
    for i, ax in enumerate(axes.flat):
        if i < len(images)*3:
            # Plot image.
            if count2==0:
                ax.imshow(np.uint8(images[count1]),interpolation=interpolation)
                count2+= 1
            elif count2==1:
                ax.imshow(np.uint8(images_right[count1]),interpolation=interpolation)
                count2+= 1
            else:
                ax.imshow(np.uint8(cls_true[count1]),interpolation=interpolation)
                count2=0
                count1+=1
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename,dpi=100)
    plt.show()
    
