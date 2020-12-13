import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
import os
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

def loadData(HRSS):
    data_path = os.path.join(os.getcwd(), 'data')
    if HRSS == 'IP':
        data = sio.loadmat(os.path.join(data_path,
                                        'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path,
                                          'Indian_pines_gt.mat'))['indian_pines_gt']
    if HRSS == 'PaU':
        data = sio.loadmat(os.path.join(data_path,
                                        'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path,
                                          'PaviaU_gt.mat'))['paviaU_gt']
    if HRSS == 'trainindexSC':
        data = sio.loadmat(os.path.join(data_path,
                                    'Indian_pines.mat'))['indian_pines']
        labels = sio.loadmat(os.path.join(data_path,
                                      'Indian_pines_gt.mat'))['indian_pines_gt']
    if HRSS == 'Selina':
        data = sio.loadmat(os.path.join(data_path,
                                        'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path,
                                          'Salinas_gt.mat'))['salinas_gt']

    return data, labels

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin,
                     X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] +
         y_offset, :] = X
    return newX

def createPatches(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize,
                            windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c -
                                margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def datasetselectedbyinformationentropy(HRSS = 'IP',windowsize = 7,CM = 'kmeans',numcomponents = 30,obwindowsize = 11): #根据信息熵图选取样本
    data_path = os.path.join(os.getcwd(), 'hotspot')
    y_ie = np.load(os.path.join(data_path + "/hotspot_" + HRSS + "_" + CM + "_PCA" + str(numcomponents) + '_obws'+ str(obwindowsize) + ".npy")).reshape(-1)
    X, y = loadData(HRSS)
    X, pca = applyPCA(X, numComponents=numcomponents)
    XPatches, yPatches = createPatches(X, y, windowSize=windowsize, removeZeroLabels=False)
    c, numsample = np.unique(yPatches, return_counts=True)
    c = c[1:]
    m = np.array([3,42,24,7,14,21,3,14,3,29,73,17,6,37,11,3])
    Xlist = np.arange(0, XPatches.shape[0])
    Xtuple = zip(Xlist, y_ie.reshape(-1))
    Xtuple = list(Xtuple)
    Xdict = zip(yPatches.reshape(-1), Xtuple)
    Xlist = list(Xdict)
    locationtrains = []
    locationtests = []
    for a in range(c.shape[0]):
        n = list()
        for b in range(len(Xlist)):
            if Xlist[b][0] == a + 1:
                n.append(Xlist[b][1])
        sortedX = sorted(n, key=lambda x: x[1], reverse=True)
        lines = np.linspace(0, len(n) - 1, m[a])
        lines = np.around(lines).astype(int)
        locationtrain = []
        locationtest = []
        dif = np.arange(0, len(n))
        difference = np.array(list(set(dif).difference(set(lines))))
        for i in range(lines.shape[0]):
            locationtrain.append(sortedX[lines[i]])
        for j in range(difference.shape[0]):
            locationtest.append(sortedX[difference[j]])
        trainindex = []
        testindex = []
        for i in range(len(locationtrain)):
            trainindex.append(locationtrain[i][0])
        for j in range(len(locationtest)):
            testindex.append(locationtest[j][0])
        locationtrains.extend(trainindex)
        locationtests.extend(testindex)
    return  locationtrains

def padWith2DZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] +
         y_offset] = X
    return newX

def createinitialPatches(X,windowSize=5):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWith2DZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize,
                            windowSize))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c -
                                margin:c + margin + 1]
            patchesData[patchIndex, :, :] = patch
            patchIndex = patchIndex + 1

    return patchesData

def calentropymap (y,obwindowsize):
    patches = createinitialPatches(y,obwindowsize)
    patches = patches.astype(int)
    entropymap = np.zeros(y.shape[0]*y.shape[1])

    for r in range(patches.shape[0]):
        c,d = np.unique(patches[r],return_counts=True)
        entropy = 0.0
        if c[0] == 0 and c.shape[0] == 1:
            entropy = 0.0
        else:
            if c[0] == 0:
                sum = obwindowsize**2- d[0]
            else:
                sum = obwindowsize**2
            if c[0] == 0:
                for m in range(1,c.shape[0]):
                    p = d[m]/sum
                    logp = np.log2(p)
                    entropy -= logp
            else:
                for m in range(c.shape[0]):
                    p = d[m]/sum
                    logp = np.log2(p)
                    entropy -= logp
        entropymap[r] = entropy

    entropymap = entropymap.reshape(y.shape[0],y.shape[1])
    return entropymap

def Pseudolabels(HRSS,CM,numcomponents,randomseeds,n_clusters,times):
    X, y = loadData(HRSS)
    X, pca = applyPCA(X, numComponents=numcomponents)
    Xr = X.reshape(X.shape[0]*X.shape[1], numcomponents)
    if CM == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=randomseeds,n_jobs=-1).fit(Xr).labels_.reshape(X.shape[0], X.shape[1])
    if CM == 'SC':
        clustering = SpectralClustering( n_jobs=-1).fit(Xr).labels_.reshape(X.shape[0], X.shape[1])
    if CM == 'HC':
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(Xr).labels_.reshape(X.shape[0], X.shape[1])
    pseudolabels = np.array([x + 1 for x in clustering])
    data_path = os.path.join(os.getcwd(), 'pseudolabels')
    with open(os.path.join(data_path, "pseudolabels_" + HRSS + "_" + CM + "_PCA" + str(numcomponents)) +'_'+str(times)+ '_' + str(n_clusters) +
              ".npy", 'bw') as outfile:
        np.save(outfile, pseudolabels)

def entromap(HRSS = 'IP',CM = 'kmeans',numcomponents = 30,obwindowsize = 11,n_clusters =16,times = 0):
    data_path = os.path.join(os.getcwd(), 'pseudolabels')
    pseudolabels = np.load(data_path +  "/pseudolabels_" + HRSS + "_" + CM + "_PCA" + str(numcomponents) + '_'+str(times)+ '_' + str(n_clusters) + ".npy")
    entropymap = calentropymap(pseudolabels,obwindowsize)
    data_path = os.path.join(os.getcwd(), 'hotspot')
    with open(os.path.join(data_path, "hotspot_" + HRSS + "_" + CM + "_PCA" + str(numcomponents) + '_obws'+ str(obwindowsize)) +
              ".npy", 'bw') as outfile:
        np.save(outfile, entropymap)

if __name__ == '__main__':
    HRSS = 'IP'
    numcomponents = 30
    n_clusters = 16
    obwindowsize  = 11
    randomseeds = np.random.randint(1,10000,10)
    datapath = os.getcwd()
    f = open(datapath + "/seeds" + ".txt", "a")
    for seed in randomseeds:
        f.write(str(seed) + ',')
    f.close()
    CM = 'kmeans'
    ITR = 1
    for i in range(ITR):
        Pseudolabels(HRSS, CM, numcomponents, randomseeds[i], n_clusters, i)
        entromap(HRSS, CM, numcomponents, obwindowsize, n_clusters, i)
        datasetselectedbyinformationentropy()

