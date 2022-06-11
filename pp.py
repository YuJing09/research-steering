from tools.lib.logreader import LogReader
from tools.lib.framereader import FrameReader
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import h5py
from common.transformations.camera import  eon_intrinsics
from common.transformations.model import get_camera_frame_from_model_frame




path0='//home//yue//Chunk_1//'

dirr=os.listdir(path0) 
print dirr
path=path0+'b0c9d2329ad1606b_2018-08-01--21-13-49//'
dirr=os.listdir(path)

path1=[path+i+'//' for i in dirr]
class CalibrationTransformsForWarpMatrix(object):
  def __init__(self, model_to_full_frame, K, E):
    self._model_to_full_frame = model_to_full_frame
    self._K = K
    self._E = E

  
  def model_to_bb(self):
      
      return _FULL_FRAME_TO_BB.dot(self._model_to_full_frame)

  
  def model_to_full_frame(self):
    return self._model_to_full_frame

  
  def car_to_model(self):
    return np.linalg.inv(self._model_to_full_frame).dot(self._K).dot(
      self._E[:, [0, 1, 3]])

  
  def car_to_bb(self):
    return _FULL_FRAME_TO_BB.dot(self._K).dot(self._E[:, [0, 1, 3]])
_BB_OFFSET = 0,0
_BB_SCALE = 1164/640.
_BB_TO_FULL_FRAME = np.asarray([
    [_BB_SCALE, 0., _BB_OFFSET[0]],
    [0., _BB_SCALE, _BB_OFFSET[1]],
    [0., 0.,   1.]])
_FULL_FRAME_TO_BB = np.linalg.inv(_BB_TO_FULL_FRAME)
#fr.frame_count
def mkcamera(path):
    
    intrinsic_matrix=eon_intrinsics
    dirr=os.listdir(path)
    for n in dirr:
        path1=path+n+'//'
        fr=FrameReader(path1+'video.hevc')
        log=h5py.File(path1+"log.h5", "r")
        print path1
    	with h5py.File(path1+'camera.h5','w') as f:
     		f.create_dataset('camera',(fr.frame_count,480,640,3))
        	f.create_dataset('X',(fr.frame_count,160,320,3))
     		for i in range(fr.frame_count):
                	extrinsic_matrix=log['extrinsicmatrix'][i]
                	ke = intrinsic_matrix.dot(extrinsic_matrix)
                	warp_matrix = get_camera_frame_from_model_frame(ke)
			img=fr.get(i,pix_fmt='rgb24')[0]
                	calibration = CalibrationTransformsForWarpMatrix(warp_matrix, intrinsic_matrix, extrinsic_matrix)
     			imgf=cv2.warpAffine(img, (_BB_TO_FULL_FRAME)[:2],
        		(640, 480), dst=img, flags=cv2.WARP_INVERSE_MAP)
                	imgm = cv2.warpAffine(imgf,(calibration.model_to_bb())[:2], (320,160), flags=cv2.WARP_INVERSE_MAP)
        		print i
        		f['camera'][i]=imgf
                	f['X'][i]=imgm
                
if __name__=='__main__':
	#for k in range(len(path1)):
        print path
        a=input('make sure:')
        if a=='y':
     
    	   mkcamera(path)
        
    	#print k

	#mk.mkframetime(path)
	#mk.frameangle(path1)

