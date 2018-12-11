#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:59:44 2017

@author: sorenc
"""
import numpy as np
import matplotlib.pyplot as pp


from scipy.misc import imsave as imsave
import  scipy.ndimage.morphology
def montage(stackin):     #stack must be 3D
    indims=stackin.shape
    

    
    xlen=indims[0]
    ylen=indims[1]
    numims=indims[2]
    
    sqval=np.sqrt(numims);
    sqval=int(np.ceil(sqval));
    rownum=sqval;
    if (sqval-1)*sqval>=numims:
        rownum=sqval-1
    rownum=int(rownum)

    if len(indims)>3:
        numchannels=indims[3]
        mont=np.zeros( (xlen*rownum,ylen*sqval,numchannels),dtype=stackin.dtype)
    else:
        mont=np.zeros( (xlen*rownum,ylen*sqval),dtype=stackin.dtype)
        numchannels=1
    
  
    for nim in range(numims):
        print nim
        colstart=int( ((np.floor((nim)/sqval)) )*ylen)
        rowstart=np.remainder(nim,sqval)*xlen
        
        rowend=rowstart+ylen
        colend=colstart+xlen
        print str(colstart) + " " + str(colend)
#disp([num2str(rowstart) ' ' num2str(colstart) ' ' num2str(rowend) ' ' num2str(colend)]);
        print str(nim) + " was pasted into X,Y: "  + " " + str(rowstart) + " " + str(rowend) + " "+ str(colstart) + " " + str(colend) 
        
        if numchannels>1:
            mont[colstart:colend,rowstart:rowend,:]=stackin[:,:,nim,:]
        else:
            mont[colstart:colend,rowstart:rowend]=stackin[:,:,nim]

    return mont


def get_edges(uint8mask):
    
    erodedarr=uint8mask*0
    for islice in range(uint8mask.shape[2]):
        erodedarr[:,:,islice]=scipy.ndimage.morphology.binary_erosion(uint8mask[:,:,islice],iterations=1)   
    
    edges=(uint8mask-erodedarr).astype(np.uint8).copy()
    return edges

def get_edges2d(uint8mask):
    
    erodedarr=uint8mask*0
    
    erodedarr[:,:]=scipy.ndimage.morphology.binary_erosion(uint8mask[:,:],iterations=1)   
    
    edges=(uint8mask-erodedarr).astype(np.uint8).copy()
    return edges

def imrescale(imin,A,B):
    fromrange=np.array(A,dtype=np.float)
    torange=np.array(B,dtype=np.float)
    if fromrange.size==0:
        fromrange=np.zeros(2,dtype=np.float)
        fromrange[0]=np.min(imin)
        fromrange[1]=np.max(imin)


    imout=(imin-fromrange[0])/(fromrange[1]-fromrange[0]) #now 0-1


    imout=imout*(torange[1]-torange[0])+torange[0];


#clamp
    imout[imout>torange[1]]=torange[1]
    imout[imout<torange[0]]=torange[0]
    return imout

def rgbmaskonrgb(rgbimg,maskimg):
    logic=np.sum(maskimg,2)>0
    
    logic3D=np.stack( (logic,logic,logic),2)
    rgbimg[logic3D]=maskimg[logic3D]
    return rgbimg


def stack2rgbmont(stack,rgb):
     mont=montage(stack)  
     return np.stack((mont*rgb[0],mont*rgb[1],mont*rgb[2]),axis=2)
    
if __name__ == "__main__":
    stack=np.zeros((128,128,10),np.float)
    for k in range(10):
        stack[:,:,k]=k*255.0/9
    stack=stack.astype(np.uint8)
    mont=montage(stack)
    pp.figure
    pp.imshow(mont[:,:])
    pp.show()
    
    
    datafolder="/home/sorenc/octopus_data/"
    ctloc=datafolder + "scct_unsmooth_res.mnc"
#==============================================================================
#     ctimg=readITK.readITKasF(ctloc)
#     
#     ctarr=(itk.GetArrayFromImage(ctimg).transpose( (1,2,0)))
#     ctarr=ctarr[:,:,::10]
#     ctarr_r=imrescale(ctarr,[0,150],[0,1])
#     rgbmont=stack2rgbmont(ctarr_r,np.ones(3,np.float))
#     imsave("/home/sorenc/test.png",rgbmont)
#     mont=montage(ctarr)
#     pp.imshow(mont[:,:])
#     pp.gca().invert_yaxis()
#     
#     mont_res=imrescale(mont,[],[0, 1])
#     pp.imshow(mont_res[:,:])
#     pp.colorbar()
#==============================================================================
    
    
    
    