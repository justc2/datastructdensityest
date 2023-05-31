import distbnnn
import numpy as np
import pickle

def loadDataOld(fname='data/final_ip_distributions'):
    with open(fname, 'rb') as handle:
        distributions = pickle.load(handle)
    source = [distributions[i] for i in range(len(distributions)) if i % 2 == 0]
    dest = [distributions[i] for i in range(len(distributions)) if i % 2 == 1]
    return source, dest

def loadData(fname='data/ip_distributions_seven_mins'):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)
    
def TV(A, B):
    if type(A) == np.ndarray:
        return np.sum(np.abs(A - B))/2
    else:
        return sum([A[key] - B.get(key,0) for key in A if A[key] > B.get(key, 0)])

def sampleDistbn(distbn, numSamples, sparse=True):
    if sparse:
        keys = np.array(list(distbn.keys()))
        probs = np.array(list(distbn.values()))
        return np.random.choice(keys, size=numSamples, replace=True, p=probs)
    else:
        return np.random.choice(len(distbn), size=numSamples, replace=True, p=distbn)

def runExperimentOld(data, queries, sparse=True, queryNames=None, numSamples=100, fastParam=20, nAllPairs=10):
    # print('Preprocessing distributions')
    if sparse:
        distbns = distbnnn.DistbnNN(K=len(data), sparse=True, preprocessScheffe=False)
    else:
        distbns = distbnnn.DistbnNN(K=data.shape[0], N=data.shape[1], sparse=False, preprocessScheffe=False)
    distbns.setDistbns(data)
    print('Running tournaments')
    print('Samples: {}, fastParam: {}, nAllPairs: {}'.format(numSamples, fastParam, nAllPairs))
    for i, query in enumerate(queries):
        sample = sampleDistbn(query, numSamples, sparse=sparse)
        slowNN, slowOps = distbns.runTournament(sample, fast=False, fastParam=fastParam, nAllPairs=nAllPairs)
        fastNN, fastOps = distbns.runTournament(sample, fast=True, fastParam=fastParam, nAllPairs=nAllPairs)
        slowTV = TV(data[int(slowNN)], query)
        fastTV = TV(data[int(fastNN)], query)
        if queryNames is not None:
            print('{} (NN, TV, Ops)'.format(queryNames[i]))
        print('SlowTournament: {}, {}, {}'.format(slowNN, slowTV, slowOps))
        print('FastTournament: {}, {}, {}'.format(fastNN, fastTV, fastOps))
        print()

def runExperiment(data, numQueries, sparse=False, numSamples=100, fastParam=20, nAllPairs=10):
    print('Samples: {}, fastParam: {}, nAllPairs: {}'.format(numSamples, fastParam, nAllPairs))
    if sparse:
        distbns = distbnnn.DistbnNN(K=2048, sparse=True, preprocessScheffe=False)
    else:
        distbns = distbnnn.DistbnNN(K=2048, N=data.shape[1], sparse=False, preprocessScheffe=False)
    for i in range(numQueries):
        query = 2048 + i
        if sparse:
            distbns.setDistbns(data[i:query])
        else:
            distbns.setDistbns(data[i:query, :])
        sample = sampleDistbn(data[query], numSamples, sparse=sparse)
        slowNN, slowOps = distbns.runTournament(sample, fast=False, fastParam=fastParam, nAllPairs=nAllPairs)
        fastNN, fastOps = distbns.runTournament(sample, fast=True, fastParam=fastParam, nAllPairs=nAllPairs)
        slowNN = int(slowNN) + i
        fastNN = int(fastNN) + i
        slowTV = TV(data[int(slowNN)], data[query])
        fastTV = TV(data[int(fastNN)], data[query])
        print('{} (NN, TV, Ops)'.format(query))
        print('SlowTournament: {}, {}, {}'.format(slowNN, slowTV, slowOps))
        print('FastTournament: {}, {}, {}'.format(fastNN, fastTV, fastOps))
        print()

def saveScheffe(data):
    distbns = distbnnn.DistbnNN(K=len(data), sparse=True)
    distbns.setDistbns(data)
    outfile = open('data/source_scheffe4096.pkl', 'w')
    pickle.dump(distbns.scheffeSets, outfile)

def loadScheffe():
    return pickle.load('data/source_scheffe4096.pkl')

def estimateAverageTVDist(data, numPairSamples=1000):
    K = len(data)
    pairSamples = np.random.choice(K, size=2*numPairSamples, replace=True)
    TVs = np.zeros(numPairSamples)
    for i in range(numPairSamples):
        TVs[i] = TV(data[pairSamples[2*i]], data[pairSamples[2*i + 1]])
    print(np.mean(TVs))
    print(np.std(TVs))

def getTVDists(data, query, verbose=True):
    TVs = np.zeros(len(data))
    for i, distbn in enumerate(data):
        TVs[i] = TV(query, distbn)
    if verbose:
        print(np.argmin(TVs))
        print(np.min(TVs))
        print(np.mean(TVs))
        print(np.mean(TVs[1000]))
    return TVs
    # print(np.std(TVs))

def test1():
    data = np.load('data/ip_sevenmin.npy')
    print(data.shape)
    runExperiment(data, 100, numSamples=100, fastParam=10, nAllPairs=0)
    runExperiment(data, 100, numSamples=100, fastParam=10, nAllPairs=5)
    runExperiment(data, 100, numSamples=250, fastParam=25, nAllPairs=0)
    runExperiment(data, 100, numSamples=250, fastParam=25, nAllPairs=5)

def test2():
    data = loadData(fname='data/light_ip_sevenmin_normalized')
    print(len(data))
    runExperiment(data, 100, sparse=True, numSamples=50, fastParam=10, nAllPairs=10)

def test3():
    data = np.load('data/ip_sevenmin.npy')
    for i in range(100):
        print(np.min(getTVDists(data[i:i+2048,:], data[i+2048,:], verbose=False)))

if __name__ == '__main__':
    # test1()
    test3()