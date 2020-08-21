import numpy as np
import os, json, h5py, math, pdb, glob

# fr = h5py.File('sub_activitynet_v1-3.c3d.hdf5','r')
# dic = fr.keys()
# file = './ActivityC3D'
# if not os.path.exists(file):
# 	os.mkdir(file)

# for key in dic:
# 	file_save = h5py.File(os.path.join(file, key+'.h5'),'w')
# 	file_save['feature'] = np.array(fr[key]['c3d_features'])
# 	

def meanX(dataX):
    return np.mean(dataX,axis=0)

def pca(XMat, k):
    average = meanX(XMat) 
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)
    featValue, featVec=  np.linalg.eig(covX)
    index = np.argsort(-featValue)
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return finalData

fr = h5py.File('tacos_c3d_fc6_nonoverlap.hdf5','r')
dic = fr.keys()

file = './TACOS'
if not os.path.exists(file):
	os.mkdir(file)

for key in dic:
	print(key)
	file_save= np.array(fr[key]['c3d_fc6_features'])
	np.save(os.path.join(file,key+'.npy'), file_save)