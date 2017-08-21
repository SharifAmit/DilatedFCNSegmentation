import numpy as np
from PIL import Image
import os
import sys
import caffe

palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('2007_000480.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('deploy.prototxt', 'fcn2s-dilated-vgg19.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)
img_3d=np.zeros((out.shape[0],out.shape[1],3),dtype=np.uint8)
for x in range(img_3d.shape[0]):
    			for y in range(img_3d.shape[1]):
				if out[x][y]==0:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=0
						img_3d[x][y][1]=0
						img_3d[x][y][2]=0
				if out[x][y]==1:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=128
						img_3d[x][y][1]=0
						img_3d[x][y][2]=0
				if out[x][y]==2:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=0
						img_3d[x][y][1]=128
						img_3d[x][y][2]=0
				if out[x][y]==3:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=128
						img_3d[x][y][1]=128
						img_3d[x][y][2]=0
				if out[x][y]==4:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=0
						img_3d[x][y][1]=0
						img_3d[x][y][2]=128
				if out[x][y]==5:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=128
						img_3d[x][y][1]=0
						img_3d[x][y][2]=128
				if out[x][y]==6:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=0
						img_3d[x][y][1]=128
						img_3d[x][y][2]=128
				if out[x][y]==7:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=128
						img_3d[x][y][1]=128
						img_3d[x][y][2]=128
				if out[x][y]==8:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=64
						img_3d[x][y][1]=0
						img_3d[x][y][2]=0
				if out[x][y]==9:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=192
						img_3d[x][y][1]=0
						img_3d[x][y][2]=0
				if out[x][y]==10:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=64
						img_3d[x][y][1]=128
						img_3d[x][y][2]=0			
				if out[x][y]==11:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=192
						img_3d[x][y][1]=128
						img_3d[x][y][2]=0
				if out[x][y]==12:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=64
						img_3d[x][y][1]=0
						img_3d[x][y][2]=128			
				if out[x][y]==13:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=192
						img_3d[x][y][1]=0
						img_3d[x][y][2]=128
							
				if out[x][y]==14:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=64
						img_3d[x][y][1]=128
						img_3d[x][y][2]=128						
				if out[x][y]==15:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=192
						img_3d[x][y][1]=128
						img_3d[x][y][2]=128
				if out[x][y]==16:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=0
						img_3d[x][y][1]=64
						img_3d[x][y][2]=0
				if out[x][y]==17:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=128
						img_3d[x][y][1]=64
						img_3d[x][y][2]=0			
				if out[x][y]==18:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=0
						img_3d[x][y][1]=192
						img_3d[x][y][2]=0
							
				if out[x][y]==19:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=128
						img_3d[x][y][1]=192
						img_3d[x][y][2]=0						
				if out[x][y]==20:
					for z in range(img_3d.shape[2]):
						img_3d[x][y][0]=0
						img_3d[x][y][1]=64
						img_3d[x][y][2]=128

img = Image.fromarray(img_3d)
img.save('ourmodelfinal.png')





