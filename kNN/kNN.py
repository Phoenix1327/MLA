#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Administrator'
import numpy as np
import operator
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    featureSize = dataSet.shape[1]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet #dataSetSize * featureSize
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#dataSetSize * 1
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort(axis=None)

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def label2int(label):
    if label == 'largeDoses':
        return 3
    elif label == 'smallDoses':
        return 2
    else:
        return 1

def file2matrix(filename):
    fr = open(filename)
    dataLines = fr.readlines()
    dataSize = len(dataLines)
    returnMat = np.zeros((dataSize, 3))
    classLabelVector = []
    index = 0
    for line in dataLines:
        line = line.strip()
        linelist = line.split('\t')
        returnMat[index,:] = linelist[0:3]
        classLabelVector.append(label2int(linelist[-1]))
        index += 1

    return returnMat, classLabelVector

def plt3DData(mat, labelvector, norm=False):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for c, m, classidx, label in [('r', 'o', 1, 'didntLike'), ('b', '^', 2, 'smallDoses'), ('g', 's', 3, 'largeDoses')]:
        idx = [i for i,a in enumerate(labelvector) if a==classidx]
        features = mat[idx][:]
        xs = features[:,0]
        ys = features[:,1]
        zs = features[:,2]
        if norm:
            xs = (xs - xs.min(0))/(xs.max(0) - xs.min(0))
            ys = (ys - ys.min(0))/(ys.max(0) - ys.min(0))
            zs = (zs - zs.min(0))/(zs.max(0) - zs.min(0))
        ax.scatter(xs, ys, zs, c=c, marker=m, label=c+': '+label)


    ax.set_xlabel('每年获得的飞行常客里程数')
    ax.set_ylabel('每周消费的冰激凌公升数')
    ax.set_zlabel('玩视频游戏所耗时间百分比')

    plt.legend()
    plt.grid(True)
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    dataSize = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (dataSize,1))
    normDataSet = normDataSet / np.tile(ranges, (dataSize,1))
    return normDataSet, ranges, minVals

def datingClassTest(k, hoRatio, datingDataMat, datingLabels):
    normMat, ranges, minVals = autoNorm(datingDataMat)
    dataSize = normMat.shape[0]
    numTestVecs = int(dataSize * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        Result = classify0(normMat[i,:], normMat[numTestVecs:,:],\
                           datingLabels[numTestVecs:],k)
        print("the classifier came back with: %d, the real answer is: %d"\
              % (Result, datingLabels[i]))
        if not Result == datingLabels[i]:
            errorCount += 1
    print("the total error rate is: %0.2f%%" % (errorCount/float(numTestVecs)*100))


mat, labelvector = file2matrix('datingTestSet.txt')
datingClassTest(3, 0.1, mat, labelvector)
plt3DData(mat, labelvector)