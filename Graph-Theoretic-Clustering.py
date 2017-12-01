import pandas as pd
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

#Read Dataset
trainingdata = pd.read_csv('CensusIncome/CencusIncome.data.txt', header = None)
testdata = pd.read_csv('CensusIncome/CencusIncome.test.txt', header = None)
numericattributes = [0,2,4,10,11,12]
nominalattributes = [1,3,5,6,7,8,9,13]
classattributes = 14


for i in nominalattributes:
	for j in range(len(trainingdata)):
		if (trainingdata.iloc[j,i].find('?') > -1):
			trainingdata.iloc[j,i] = trainingdata.mode()[i][0]
	for j in range(len(testdata)):
		if (testdata.iloc[j,i].find('?') > -1): testdata.iloc[j,i] = trainingdata.mode()[i][0]
	
print(trainingdata.iloc[[149]])


# train = trainingdata

# trainingdata = trainingdata.as_matrix()
# testdata = testdata.as_matrix()
# distancematrix = np.zeros(shape(len(trainingdata), len(trainingdata)))



# #normalize:
# for j in numericattributes:
# 	max1 = train[j].max()
# 	min1 = train[j].min()
# 	range1 = max1 - min1

# 	for i in range(0, len(trainingdata)):
# 		if range1 != 0 :
# 			trainingdata[i][j] = float(trainingdata[i][j] - min1)/float(range1)
# 	for i in range(0, len(testdata)):
# 		if range1 != 0:
# 			testdata[i][j] = float(testdata[i][j] - min1)/float(range1)

df = pd.DataFrame(np.asarray(trainingdata))
df.to_csv("file_path.csv")



def countDistance(x, y):
	sum = 0
	for i in numericattributes:
		sum += math.pow(x[i] - y[i],2)
	for i in nominalattributes:
		if (x[i] != y[i]): sum += 1
	return math.sqrt(sum)

def buildDistanceMatrix():
	for i in range(0,len(trainingdata)):
		for j in range(0, len(trainingdata)):
			if (i >= j):
				distancematrix[i][j] = 0
			else:
				distancematrix[i][j] = countDistance(trainingdata[i], trainingdata[j])
	return distancematrix

# distancematrix = buildDistanceMatrix()
# distancematrix = csr_matrix(distancematrix)
# MST = minimum_spanning_tree(distancematrix)

# #Remove maximum distance
# idxMax = np.unravel_index(MST.argmax(), MST.shape)
# MST[idxMax] = 0

# mst = MST
# #Label connected components.
# num_graphs, labels = connected_components(mst, directed=False)

# # We should have two trees.
# assert(num_graphs == 2)

# # Use indices as node ids and group them according to their graph.
# results = [[] for i in range(max(labels) + 1)]
# for idx, label in enumerate(labels):
#     results[label].append(idx)

# print(results)
