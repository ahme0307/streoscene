import sys, os
import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from ImageDataGenerator import transform_matrix_offset_center
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import apply_transform, Iterator,random_channel_shift, flip_axis
from keras.preprocessing.image import Iterator,random_channel_shift
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import random
import pdb
from skimage.io import imsave, imread
from skimage.transform import rotate
from skimage import transform
from skimage.transform import resize
from  params import * 
import json
import math
#import matplotlib.pyplot as plt

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)
def RandomLight(img,img_right):
    lights = random.choice(["Rfilter","Rbright","Rcontr", "RSat","RhueSat"])
    #print(lights)
    if lights=="Rfilter":
        alpha = 0.5 * random.uniform(0, 1)
        kernel = np.ones((3, 3), np.float32)/9 * 0.2
        colored = img[..., :3]
        colored = alpha * cv2.filter2D(colored, -1, kernel) + (1-alpha) * colored
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[..., :3] = clip(colored, dtype, maxval)
        
        #right image
        colored = img_right[..., :3]
        colored = alpha * cv2.filter2D(colored, -1, kernel) + (1-alpha) * colored
        maxval = np.max(img_right[..., :3])
        dtype = img_right.dtype
        img_right[..., :3] = clip(colored, dtype, maxval)
    if lights=="Rbright":
        alpha = 1.0 + 0.1*random.uniform(-1, 1)
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[..., :3] = clip(alpha * img[...,:3], dtype, maxval)
        #right image
        maxval = np.max(img_right[..., :3])
        dtype = img_right.dtype
        img_right[..., :3] = clip(alpha * img_right[...,:3], dtype, maxval)
        
        
    if lights=="Rcontr":
        alpha = 1.0 + 0.1*random.uniform(-1, 1)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)	
        #right image
        gray = cv2.cvtColor(img_right[:, :, :3], cv2.COLOR_BGR2GRAY)
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        maxval = np.max(img_right[..., :3])
        dtype = img.dtype
        img_right[:, :, :3] = clip(alpha * img_right[:, :, :3] + gray, dtype, maxval)	
        
    if lights=="RSat":
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        alpha = 1.0 + random.uniform(-0.1, 0.1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img[..., :3] = alpha * img[..., :3] + (1.0 - alpha) * gray
        img[..., :3] = clip(img[..., :3], dtype, maxval)
        
        #righ image
        maxval = np.max(img_right[..., :3])
        dtype = img_right.dtype
        alpha = 1.0 + random.uniform(-0.1, 0.1)
        gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_right[..., :3] = alpha * img_right[..., :3] + (1.0 - alpha) * gray
        img_right[..., :3] = clip(img_right[..., :3], dtype, maxval)
    if lights=="RhueSat":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        hue_shift = np.random.uniform(-25,25)
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(-25,25)
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(-25, 25)
        v = cv2.add(v, val_shift)
        img = cv2.merge((h, s, v))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        #right image
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_right)
        h = cv2.add(h, hue_shift)
        s = cv2.add(s, sat_shift)
        v = cv2.add(v, val_shift)
        img_right = cv2.merge((h, s, v))
        img_right = cv2.cvtColor(img_right, cv2.COLOR_HSV2BGR)
        
    return img,img_right

def perspectivedist(img,img_right,img_mask, flag='all'):
    if flag=='all':           
            magnitude=3
           # pdb.set_trace()
            rw=img.shape[0]
            cl=img.shape[1]
            #x = random.randrange(50, 200)
            #nonzeromask=(img_mask>0).nonzero()
            #nonzeroy = np.array(nonzeromask[0])
            #nonzerox = np.array(nonzeromask[1])
            #bbox = (( np.maximum(np.min(nonzerox)-x,0),  np.maximum(np.min(nonzeroy)-x,0)), (np.minimum(np.max(nonzerox)+x,cl),  np.minimum(np.max(nonzeroy)+x,rw)))
            #pdb.set_trace()
          #  img=img[bbox[0][1]:(bbox[1][1]),bbox[0][0]:(bbox[1][0])]
           # img_mask=img_mask[bbox[0][1]:(bbox[1][1]),bbox[0][0]:(bbox[1][0])]
            skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
            w, h,_ = img.shape
            x1 = 0
            x2 = h
            y1 = 0
            y2 = w

            original_plane =  np.array([[(y1, x1), (y2, x1), (y2, x2), (y1, x2)]], dtype=np.float32)

            max_skew_amount = max(w, h)
            max_skew_amount = int(math.ceil(max_skew_amount *magnitude))
            skew_amount = random.randint(1, max_skew_amount)
            if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":
                if skew == "TILT":
                    skew_direction = random.randint(0, 3)
                elif skew == "TILT_LEFT_RIGHT":
                    skew_direction = random.randint(0, 1)
                elif skew == "TILT_TOP_BOTTOM":
                    skew_direction = random.randint(2, 3)

                if skew_direction == 0:
                    # Left Tilt
                    new_plane = np.array([(y1, x1 - skew_amount),  # Top Left
                                 (y2, x1),                # Top Right
                                 (y2, x2),                # Bottom Right
                                 (y1, x2 + skew_amount)], dtype=np.float32)  # Bottom Left
                elif skew_direction == 1:
                    # Right Tilt
                    new_plane = np.array([(y1, x1),                # Top Left
                                 (y2, x1 - skew_amount),  # Top Right
                                 (y2, x2 + skew_amount),  # Bottom Right
                                 (y1, x2)],dtype=np.float32)                # Bottom Left
                elif skew_direction == 2:
                    # Forward Tilt
                    new_plane = np.array([(y1 - skew_amount, x1),  # Top Left
                                 (y2 + skew_amount, x1),  # Top Right
                                 (y2, x2),                # Bottom Right
                                 (y1, x2)], dtype=np.float32)                # Bottom Left
                elif skew_direction == 3:
                    # Backward Tilt
                    new_plane = np.array([(y1, x1),                # Top Left
                                 (y2, x1),                # Top Right
                                 (y2 + skew_amount, x2),  # Bottom Right
                                 (y1 - skew_amount, x2)], dtype=np.float32)  # Bottom Left

            if skew == "CORNER":
                skew_direction = random.randint(0, 7)

                if skew_direction == 0:
                    # Skew possibility 0
                    new_plane = np.array([(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)], dtype=np.float32)
                elif skew_direction == 1:
                    # Skew possibility 1
                    new_plane = np.array([(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)], dtype=np.float32)
                elif skew_direction == 2:
                    # Skew possibility 2
                    new_plane = np.array([(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)],dtype=np.float32)
                elif skew_direction == 3:
                    # Skew possibility 3
                    new_plane = np.array([(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)], dtype=np.float32)
                elif skew_direction == 4:
                    # Skew possibility 4
                    new_plane = np.array([(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)], dtype=np.float32)
                elif skew_direction == 5:
                    # Skew possibility 5
                    new_plane = np.array([(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)], dtype=np.float32)
                elif skew_direction == 6:
                    # Skew possibility 6
                    new_plane = np.array([(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)],dtype=np.float32)
                elif skew_direction == 7:
                    # Skew possibility 7
                    new_plane =np.array([(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)], dtype=np.float32)
           # pdb.set_trace()
            perspective_matrix = cv2.getPerspectiveTransform(original_plane, new_plane)
            img = cv2.warpPerspective(img, perspective_matrix,
                                     (img.shape[1], img.shape[0]),
                                     flags = cv2.INTER_LINEAR)
            img_right = cv2.warpPerspective(img_right, perspective_matrix,
                                     (img.shape[1], img.shape[0]),
                                     flags = cv2.INTER_LINEAR)
            img_mask = cv2.warpPerspective(img_mask, perspective_matrix,
                                     (img.shape[1], img.shape[0]),
                                     flags = cv2.INTER_LINEAR)
    return img, img_right, img_mask
def apply_clahe(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img
def add_gaussian_noise(X_imgs):

    #pdb.set_trace()
    row, col,_= X_imgs.shape
    #X_imgs=X_imgs/255
    X_imgs = X_imgs.astype(np.float32)
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
    gaussian_img = cv2.addWeighted(X_imgs, 0.75, 0.25 * gaussian, 0.25, 0)
    gaussian_img = np.array(gaussian_img, dtype = np.uint8)
    return gaussian_img
def random_affine(img,img_right,img_mask):
    flat_sum_mask=sum(img_mask.flatten())
    (row,col,_)=img_mask.shape
    angle=shear_deg=0
    zoom=1
    center_shift   = np.array((1000, 1000)) / 2. - 0.5
    tform_center   = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)
    big_img=np.zeros((1000,1000,3), dtype=np.uint8)
    big_img_right=np.zeros((1000,1000,3), dtype=np.uint8)
    big_mask=np.zeros((1000,1000,3), dtype=np.uint8)
    big_img[190:(190+row),144:(144+col)]=img
    
    big_img_right[190:(190+row),144:(144+col)]=img_right
    big_mask[190:(190+row),144:(144+col)]=img_mask
    affine = random.choice(["rotate", "zoom", "shear"])
    if affine == "rotate":
        angle= random.uniform(-90, 90)
    if affine == "zoom":
        zoom = random.uniform(0.5, 1.5)
    if affine=="shear":
        shear_deg = random.uniform(-5, 5)    
   # pdb.set_trace()
    tform_aug = transform.AffineTransform(rotation = np.deg2rad(angle),
                                              scale =(1/zoom, 1/zoom),
                                              shear = np.deg2rad(shear_deg),
                                              translation = (0, 0))
    tform = tform_center + tform_aug + tform_uncenter
                   # pdb.set_trace()
    img_tr=transform.warp((big_img), tform)
    img_tr_right=transform.warp((big_img_right), tform)
    mask_tr=transform.warp((big_mask), tform)
                   # pdb.set_trace()
    masktemp =  cv2.cvtColor((img_tr*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)>20
    img_tr=img_tr[np.ix_(masktemp.any(1),masktemp.any(0))]
    mask_tr = mask_tr[np.ix_(masktemp.any(1),masktemp.any(0))]
    img_tr_right = img_tr_right[np.ix_(masktemp.any(1),masktemp.any(0))]
    return (img_tr*255).astype(np.uint8),(img_tr_right*255).astype(np.uint8),(mask_tr*255).astype(np.uint8)    


class CustomNumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='th'):
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.training=image_data_generator.training
        self.img_rows=image_data_generator.netparams.img_rows
        self.img_cols=image_data_generator.netparams.img_cols
        with open('labels_2017.json') as json_file:
            self.Data = json.load(json_file)
            #pdb.set_trace()
        super(CustomNumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)
    def _get_batches_of_transformed_samples(self, index_array):
       # pdb.set_trace()
        batch_x_right = np.zeros((len(index_array),self.img_rows,self.img_cols,3), dtype=np.float32)
        batch_x_left = np.zeros((len(index_array),self.img_rows,self.img_cols,3), dtype=np.float32)
        if self.training:
            if  self.image_data_generator.netparams.task=='all':
                ch_num=11
            elif  self.image_data_generator.netparams.task=='binary':
                ch_num=1
            elif  self.image_data_generator.netparams.task=='parts':
                ch_num=3
            elif  self.image_data_generator.netparams.task=='instrument':
                ch_num=7                
        else:
            ch_num=3
            
        batch_y=np.zeros((len(index_array), self.img_rows,self.img_cols,ch_num), dtype=np.float32)    
        infos=[]
        for i, j in enumerate(index_array):
            #pdb.set_trace()
            x_left = imread(self.X[j][0])
            x_right =imread(self.X[j][1])
            y1 =imread(self.y[j])
            y1 = y1[...,[1,2,0]]
            #print(j)
            #pdb.set_trace()
            infos.append((self.X[j][0], x_left.shape))    
            
            _x_left, _x_right, _y1 = self.image_data_generator.random_transform(x_left.astype(np.uint8), x_right.astype(np.uint8),y1.astype(np.uint8),self.Data)
            batch_x_left[i]=_x_left
            batch_x_right[i]=_x_right
            batch_y[i]=_y1
            #inf_temp=[]
            #inf_temp.append()
           # inf_temp.append()
           
          #  infos.append(
         #   pdb.set_trace()
        batch_y=np.reshape(batch_y,(-1,self.img_rows,self.img_cols,ch_num))
        return batch_x_left,batch_x_right,batch_y,infos


    def next(self):
        with self.lock:
            index_array = next(self.index_generator)  
            #print(index_array)
        return self._get_batches_of_transformed_samples(index_array)
def convert_gray(data,im, tasktype):
    #pdb.set_trace()
    
    #np.shape(self.Data['instrument'])
    if tasktype.task=='all':
        out = (np.zeros((im.shape[0],im.shape[1],11)) ).astype(np.uint8)
        #pdb.set_trace()
        image=np.squeeze(im[:,:,0])
        indexc=0
        for label_info,index in zip(data['instrument'],range(0,np.shape(data['instrument'])[0]+1)):
            rgb=label_info['color'][0]
            
            if rgb==0:
                continue
            temp_out = (np.zeros(im.shape[:2]) ).astype(np.uint8)
            gray_val=255 
            #pdb.set_trace()
            match_pxls = np.where(image == rgb)
            temp_out[match_pxls] = gray_val
            out[:,:,index-1]=temp_out
            #print(index-1)
            #print(rgb)       
        image=np.squeeze(im[:,:,1])       
        for label_info,index in zip(data['parts'],range(np.shape(data['instrument'])[0],np.shape(data['instrument'])[0]+np.shape(data['parts'])[0])):
            rgb=label_info['color'][1]
            #pdb.set_trace()
            if rgb==0:
                continue
            temp_out = (np.zeros(im.shape[:2]) ).astype(np.uint8)
            gray_val=255        
            match_pxls = np.where(image == rgb)
            temp_out[match_pxls] = gray_val
            out[:,:,index-1]=temp_out
            #print(index-1)
            #print(rgb)
        #pdb.set_trace()
        out[:,:,index]=np.squeeze(im[:,:,2])
        #print(index)
    #pdb.set_trace()
    if tasktype.task=='binary':
        out = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)   
        out[:,:]=np.squeeze(im[:,:,2])
    if tasktype.task=='instrument':
        out = (np.zeros((im.shape[0],im.shape[1],np.shape(data['instrument'])[0]-1))).astype(np.uint8)
        #pdb.set_trace()
        image=np.squeeze(im[:,:,0])
        indexc=0
        for label_info,index in zip(data['instrument'],range(0,np.shape(data['instrument'])[0]+1)):
            rgb=label_info['color'][0]
            #pdb.set_trace()
            if rgb==0:
                continue
            temp_out = (np.zeros(im.shape[:2]) ).astype(np.uint8)
            gray_val=255 
        
            match_pxls = np.where((image == rgb))
            temp_out[match_pxls] = gray_val
            out[:,:,index-1]=temp_out
    if tasktype.task=='parts':
        out = (np.zeros((im.shape[0],im.shape[1],np.shape(data['parts'])[0])) ).astype(np.uint8)
        #pdb.set_trace()
        image=np.squeeze(im[:,:,1])
        indexc=0
        for label_info,index in zip(data['parts'],range(0,np.shape(data['parts'])[0])):
            rgb=label_info['color'][1]
            #pdb.set_trace()
            if rgb==0:
                continue
            temp_out = (np.zeros(im.shape[:2]) ).astype(np.uint8)
            gray_val=255 
        
            match_pxls = np.where(image == rgb)
            temp_out[match_pxls] = gray_val
            out[:,:,index]=temp_out
    return out.astype(np.uint8)
def convert_color(data,im, tasktype):
   # pdb.set_trace()
    im=np.squeeze(im)
    if tasktype.task=='all':
        out1 = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        out2 = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        out3 = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        for label_info,index in zip(data['instrument'],range(0,np.shape(data['instrument'])[0]+1)):
            rgb=label_info['color'][0]
            if np.sum(rgb)==0:
                continue
            temp=im[:,:,index-1]
            temp=temp.astype(np.float)
            #temp =cv2.resize(temp,(224,224),interpolation=cv2.INTER_CUBIC)
            match_pxls = np.where(temp > 0.2)
            out1[match_pxls] = rgb
            
        for label_info,index in zip(data['parts'],range(np.shape(data['instrument'])[0],np.shape(data['instrument'])[0]+np.shape(data['parts'])[0])):
            rgb=label_info['color'][1]
            #pdb.set_trace()
            if np.sum(rgb)==0:
                continue
            temp=im[:,:,index-1]
            #print(index-1)
            temp=temp.astype(np.float)
            #temp =cv2.resize(temp,(224,224),interpolation=cv2.INTER_CUBIC)
            match_pxls = np.where(temp > 0.2)
            out2[match_pxls] = rgb
        out3=(im[:,:,index]>0.2)*255
        out=np.dstack((out1,out2,out3))
        #pdb.set_trace()
    if tasktype.task=='binary':
        out = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)  
        out=(im>0.2)*255
    if tasktype.task=='parts':
        out = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        for label_info,index in zip(data['parts'],range(0,np.shape(data['parts'])[0])):
            rgb=label_info['color'][1]
            if np.sum(rgb)==0:
                continue
            temp=im[:,:,index]
            temp=temp.astype(np.float)
            temp =cv2.resize(temp,(224,224),interpolation=cv2.INTER_CUBIC)
            match_pxls = np.where(temp > 0.2)
            out[match_pxls] = rgb
    if tasktype.task=='instrument':
        out = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        for label_info,index in zip(data['instrument'],range(0,np.shape(data['instrument'])[0])):
            rgb=label_info['color'][0]
            if np.sum(rgb)==0:
                continue
            temp=im[:,:,index-1]
            temp=temp.astype(np.float)
            temp =cv2.resize(temp,(224,224),interpolation=cv2.INTER_CUBIC)
            match_pxls = np.where(temp > 0.2)
            out[match_pxls] = rgb
    return out.astype(np.uint8)

'''
def convert_color(data,im, tasktype):
    #pdb.set_trace()
    out = (np.zeros((im.shape[0],im.shape[1],3)) ).astype(np.uint8)
    if tasktype.task=='all':
        
        for label_info,index in zip(data,range(0,np.shape(data)[0])):
            rgb=label_info['color']
            if np.sum(rgb)==0:
                continue
            temp_out = (np.zeros(im.shape[:2]) ).astype(np.uint8)
            match_pxls = np.where(im[:,:,index] == 255)
            #pdb.set_trace()
            out[match_pxls] = rgb            
   # assert (out != 255).all(), "rounding errors or missing classes in camvid_colors"
    return out.astype(np.uint8)
'''
class CustomImageDataGenerator(object):
    def __init__(self, netparams,training):
        self.netparams = netparams
        #self.CROP = CROP
        #self.perspective = perspective
        #self.lighting = lighting
        #self.Flip =Flip
        #self.affine=affine
        #self.randcrop=randcrop
        self.training =training
        #CLAHE=True, CROP=True, perspective=True,lighting=True,Flip=True,affine=True,randcrop=True
    
    def random_transform(self, img_left,img_right,img_mask,label_data):
        image_rows = 224
        image_cols = 224
        rw=img_left.shape[0]
        cl=img_left.shape[1]
        ch=np.shape(img_left.shape)[0]
        flag_crop=None
        #pdb.set_trace()
        img_left =cv2.resize(img_left, (image_rows,image_cols))
        img_right =cv2.resize(img_right, (image_rows,image_cols))
        img_mask = cv2.resize(img_mask, (image_rows,image_cols))
        img_mask=img_mask[:,:,0:3]    
        augCh = random.choice(["CROP","PER","ORIG", "FLIP","AFFINE","ORIG","randcrop","LIGHT"])

        if self.netparams.CLAHE and  augCh=="CLAHE":
            img_left=apply_clahe(img_left)
            img_right=apply_clahe(im_right)       
            pdb.set_trace()
        if  self.netparams.perspective and augCh=="PER":
            pdb.set_trace()
            img_left,img_right,img_mask=perspectivedist(img_left,img_right,img_mask,'all')
            
        if self.netparams.affine and augCh=="AFFINE":
            #pdb.set_trace()
            img_left,img_right,img_mask=random_affine(img_left, img_right,img_mask)
            pdb.set_trace()
        if self.netparams.lighting and augCh=="LIGHT":
            img_left,img_right = RandomLight(img_left,img_right)
            pdb.set_trace()
             
        if self.netparams.Flip and augCh=="FLIP":
            pdb.set_trace()
            flHV = random.choice(["H", "V"])
            if flHV=="H":
                #pdb.set_trace()
                img_left = cv2.flip(img_left, 0 )
                img_right =cv2.flip(img_right,0)
                img_mask= cv2.flip( img_mask, 0)
                
            else:
                #pdb.set_trace()
                img_left = cv2.flip(img_left,1 )
                img_right=cv2.flip(img_right,1)
                img_mask= cv2.flip( img_mask, 1)
        if self.netparams.randcrop and augCh=='randcrop':
            pdb.set_trace()
            dx = dy = 112
            rx=random.randint(0, image_rows-dx-1)
            ry=random.randint(0, image_rows-dy-1)
            #pdb.set_trace()
            img_left = img_left[ry :ry +dy,  rx: rx+dx]
            img_right = img_right[ry :ry +dy,  rx: rx+dx]
            img_mask=img_mask[ry :ry +dy,  rx: rx+dx]
            
        img_left= cv2.resize(img_left, (image_rows,image_cols))
        img_right= cv2.resize(img_right, (image_rows,image_cols))
        img_mask =  cv2.resize(img_mask, (image_rows,image_cols))
        #temp2=img_mask
        #pdb.set_trace()
        if self.training:
            img_mask= convert_gray(label_data,img_mask,self.netparams)
        #pdb.set_trace()
        #img_mask= convert_color(label_data,img_mask,self.netparams)
        #pdb.set_trace()
        img_left = img_left.astype('float32')
        img_right = img_right.astype('float32')
        img_left/=255.
        img_right/=255.
        img_mask=img_mask.astype('float32')
        img_mask /= 255.  # scale masks to [0, 1]
        #temp= convert_color(label_data,img_mask,self.netparams)
        #pdb.set_trace()
        return np.array(img_left), np.array(img_right), np.array(img_mask)
    
    def flow(self, X, Y, batch_size, shuffle=True, seed=None):
        return CustomNumpyArrayIterator(
            X, Y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed)


    



def loaddataset():
    imgs_test = np.load('/media/a252/540/imgs_testShuffled.npy',mmap_mode='r')
    #imgs_test = np.memmap('imgs_test.npy', mode='r')
    imgs_id = np.load('/media/a252/540/imgs_mask_testShuffle.npy',mmap_mode='r')
    imgs_train = np.load('/media/a252/540/imgs_trainShuffled.npy',mmap_mode='r')
    imgs_mask_train = np.load('/media/a252/540/imgs_mask_trainShuffle.npy',mmap_mode='r')
    return imgs_test,imgs_id,
    
    
