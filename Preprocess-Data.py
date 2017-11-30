import pandas as pd
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

#Read Dataset
trainingdata = pd.read_csv('CensusIncome_Normalize/CencusIncome.data.txt', header = None)
testdata = pd.read_csv('CensusIncome_Normalize/CencusIncome.test.txt', header = None)

train = trainingdata

trainingdata = trainingdata.as_matrix()
testdata = testdata.as_matrix()

numericattributes = [0,2,4,10,11,12]
nominalattributes = [1,3,5,6,7,8,9,13]
classattributes = 14

for j in numericattributes:
	max1 = train[j].max()
	min1 = train[j].min()
	range1 = max1 - min1

	for i in range(0, len(trainingdata)):
		if range1 != 0 :
			trainingdata[i][j] = float(trainingdata[i][j] - min1)/float(range1)
	for i in range(0, len(testdata)):
		if range1 != 0:
			testdata[i][j] = float(testdata[i][j] - min1)/float(range1)

print(trainingdata)
print(testdata)

trainingdata = np.array(trainingdata)

# with open('CensusIncome_Normalize/CencusIncome.test1.txt', 'w') as f:
# 	np.savetxt(f, trainingdata, delimiter=",")