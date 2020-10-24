# Helpful Tips: https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from math import exp
import cv2
import matplotlib.pyplot as plt

def sobel(window_size):
	assert(window_size%2!=0)
	ind=window_size/2
	matx=[]
	maty=[]
	for j in range(-ind,ind+1):
		row=[]
		for i in range(-ind,ind+1):
			if (i*i+j*j)==0:
				gx_ij=0
			else:
				gx_ij=i/float(i*i+j*j)
			row.append(gx_ij)
		matx.append(row)
	for j in range(-ind,ind+1):
		row=[]
		for i in range(-ind,ind+1):
			if (i*i+j*j)==0:
				gy_ij=0
			else:
				gy_ij=j/float(i*i+j*j)
			row.append(gy_ij)
		maty.append(row)

	# matx=[[-3, 0,+3],
	# 	  [-10, 0 ,+10],
	# 	  [-3, 0,+3]]
	# maty=[[-3, -10,-3],
	# 	  [0, 0 ,0],
	# 	  [3, 10,3]]
	if window_size==3:
		mult=2
	elif window_size==5:
		mult=20
	elif window_size==7:
		mult=780

	matx=np.array(matx)*mult				
	maty=np.array(maty)*mult

	return torch.Tensor(matx), torch.Tensor(maty)

def create_window(window_size, channel):
	windowx,windowy = sobel(window_size)
	windowx,windowy= windowx.unsqueeze(0).unsqueeze(0), windowy.unsqueeze(0).unsqueeze(0)
	windowx = torch.Tensor(windowx.expand(channel,1,window_size,window_size))
	windowy = torch.Tensor(windowy.expand(channel,1,window_size,window_size))
	return windowx,windowy

def gradient(img, windowx, windowy, window_size, padding, channel):
	if channel > 1 :		# do convolutions on each channel separately and then concatenate
		gradx=torch.ones(img.shape)
		grady=torch.ones(img.shape)
		for i in range(channel):
			gradx[:,i,:,:]=F.conv2d(img[:,i,:,:].unsqueeze(0), windowx, padding=padding,groups=1).squeeze(0)   #fix the padding according to the kernel size
			grady[:,i,:,:]=F.conv2d(img[:,i,:,:].unsqueeze(0), windowy, padding=padding,groups=1).squeeze(0)
	else:
		gradx = F.conv2d(img, windowx, padding=padding,groups=1)
		grady = F.conv2d(img, windowy, padding=padding,groups=1)

	return gradx, grady

class SobelGrad(torch.nn.Module):
	def __init__(self, window_size = 3, padding= 1):
		super().__init__()
		self.window_size = window_size
		self.padding= padding
		self.channel = 1			# out channel
		self.windowx,self.windowy = create_window(window_size, self.channel)

	def forward(self, pred,label):
		(batch_size, channel, _, _) = pred.size()
		if pred.is_cuda:
			self.windowx = self.windowx.cuda(pred.get_device())
			self.windowx = self.windowx.type_as(pred)
			self.windowy = self.windowy.cuda(pred.get_device())
			self.windowy = self.windowy.type_as(pred)
			
		pred_gradx, pred_grady = gradient(pred, self.windowx, self.windowy, self.window_size, self.padding, channel)
		label_gradx, label_grady = gradient(label, self.windowx, self.windowy, self.window_size, self.padding, channel)

		return pred_gradx, pred_grady, label_gradx, label_grady

class Sobel(nn.Module):
    """Sobel layer."""

    def __init__(self):
        super(Sobel, self).__init__()
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
        sobel_filter.bias.data.zero_()
        self.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in self.sobel.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.sobel(x)
