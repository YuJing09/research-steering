import os
import h5py
import cv2
import pickle
import numpy as np
import h5py
from tools.lib.logreader import LogReader
from tools.lib.framereader import FrameReader



path0='//home//yue//Chunk_1//b0c9d2329ad1606b_2018-08-01--21-13-49//'
q=os.listdir(path0)
path=[path0+i+'//' for i in q]
def mkframetime(path):
    dirr=os.listdir(path)
    for i in (dirr):
        print i
        path1=path+i+'//'
        lr=LogReader(path1+'raw_log.bz2')
        logs=list(lr)
        frametime=[l.logMonoTime*10**-9 for l in logs if l.which=='frame']
        np.save(path1+'frametime',frametime)
def mkseqdir(path):
    dir0=os.listdir(path)
    for n,d in enumerate(dir0):
        os.rename(path+d,path+str(n+1))
def mkcameradir(path):
    dir0=os.listdir(path)
    for d in dir0:
        os.mkdir(path+d+'//camera')

def mkcamera(path):
	dir0=os.listdir(path)
        x,y,z=160,320,3
        for di in dir0:
                print di
		path0=path+di
                log=h5py.File(path0+'//'+"log.h5", "r")
                
            
        	#a=h5py.File(path0+'//camera1.h5','w')
        	#a.create_dataset('camera',(100,x,y,z))
            
        	
        	for i in range(100):
                	print i
                        extrinsic_matrix=log['extrinsicmatrix'][i]
                        intrinsic_matrix=eon_intrinsics
                        ke = eon_intrinsics.dot(extrinsic_matrix)
                        warp_matrix = get_camera_frame_from_model_frame(ke)
    
                        calibration = CalibrationTransformsForWarpMatrix(warp_matrix, intrinsic_matrix, extrinsic_matrix)               
        		img=cv2.imread(path0+'//camera//'+str(i)+'.png')
                        imgg=cv2.warpAffine(img,calibration.model_to_bb()[:2],(320,160),flags=cv2.WARP_INVERSE_MAP)
        		imgg=imgg[:,:,::-1]
        	
                        print imgg.shape
        		#a['camera'][i]=imgg
        
        	#a.close()    
	    
             	     
def pickletonp(path):
        dirr=os.listdir(path)
	for d in dirr:
	    with open(path+d+'//frame_time') as f:
            	frame_time=pickle.load(f)*10**-9
                np.save(path+d+'//frametime',frame_time)
def frameangle(path):
	for d in path:
            print d
            speed=np.load(d+'processed_log//CAN//speed//value')
            t0=np.load(d+'processed_log//CAN//speed//t') 
	    steering_angle=np.load(d+'processed_log//CAN//steering_angle//value')
            
            t=np.load(d+'processed_log//CAN//steering_angle//t')
            frametime=np.load(d+'frametime.npy')
            
            lr=LogReader(d+'raw_log.bz2')
            logs=list(lr)
            curvature=[l.controlsState.curvature for l in logs if l.which()=='controlsState']
            curvature_t=[l.logMonoTime*10**-9 for l in logs if l.which()=='controlsState']
            livecalibration=[np.asarray(l.liveCalibration.extrinsicMatrix).reshape(3, 4) for l in logs if l.which()=='liveCalibration']
            livectime=[l.logMonoTime*10**-9 for l in logs if l.which()=='liveCalibration']
            n=len(frametime)
            with h5py.File(d+'log.h5','w') as f:
                 f.create_dataset('curvature',(n,1))
                 f.create_dataset('steering_angle',(n,1))
                 f.create_dataset('speed',(n,1))
                 f.create_dataset('extrinsicmatrix',(n,3,4))
            	 for i in range(len(frametime)):
                	abs_an=np.abs(t-frametime[i])
                	abs_sp=np.abs(t0-frametime[i])
                	abs_cur=np.abs(curvature_t-frametime[i])
                	abs_ex=np.abs(livectime-frametime[i])
                        ind_an=np.argmin(abs_an)
                        ind_sp=np.argmin(abs_sp)
                        ind_cur=np.argmin(abs_cur)
                        ind_ex=np.argmin(abs_ex)
                        f['curvature'][i]=curvature[ind_cur]
                        f['extrinsicmatrix'][i]=livecalibration[ind_ex]
                        f['steering_angle'][i]=steering_angle[ind_an]
                        f['speed'][i]=speed[ind_sp]
            
            
	    
	         
if __name__ == "__main__":
	#mkcamera(path0)
        #frameangle(path)
        #mkseqdir(path0)        
        #mkcameradir(path0)
        mkframetime(path0)
        print path0
        a=input('make sure:')
        if a=='y':
        	frameangle(path)
        












        #pickletonp(path)
        
         
