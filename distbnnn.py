import numpy as np
import math

class DistbnNN():
    def __init__(self, K, N=-1, sparse=False, preprocessScheffe=True):
        self.K = K # num distributions
        self.N = N # domain size
        self.distbns = None
        self.sparse = sparse
        self.preprocess = preprocessScheffe
        self.scheffeSets = None
        if not sparse:
            assert(self.N > 0 )

    def setDistbns(self, distbns):
        self.distbns = distbns
        if not self.sparse:
            assert(distbns.shape == (self.K, self.N))
        if self.preprocess:
            self.preprocessScheffe()

    def setScheffe(self, scheffeSets):
        self.scheffeSets = scheffeSets
        self.preprocess = True

    def sampleDistbn(self, distID, numSamples):
        if not self.sparse:
            return np.random.choice(self.N, size=numSamples, replace=True, p=self.distbns[distID])
        else:
            keys = np.array(self.distbns[distID].keys())
            probs = np.array(self.distbns[distID].vals())
            return np.random.choice(keys, size=numSamples, replace=True, p=probs)
    
    def preprocessScheffe(self):
        self.scheffeSets = dict()
        for i in range(self.K):
            if i % 100 == 0:
                print(i)
            for j in range(self.K):
                if i == j:
                    continue
                if not self.sparse:
                    S = self.distbns[i] > self.distbns[j]
                    vi = np.sum(self.distbns[i][S])
                    vj = np.sum(self.distbns[j][S])
                else:
                    S = [key for key in self.distbns[i] if self.distbns[i][key] > self.distbns[j].get(key,0)]
                    vi = sum([self.distbns[i][key] for key in S])
                    vj = sum([self.distbns[j].get(key, 0) for key in S])
                self.scheffeSets[(i,j)] = (S, vi, vj)

    def scheffeTest(self, i, j, sample):
        countS = 0
        if self.preprocess:
            S, vi, vj = self.scheffeSets[(i,j)]
        else:
            if not self.sparse:
                S = self.distbns[i] > self.distbns[j]
                vi = np.sum(self.distbns[i][S])
                vj = np.sum(self.distbns[j][S])
            else:
                vi = 0
                vj = 0
                S = set()
                for key in self.distbns[i]:
                    pi = self.distbns[i][key]
                    pj = self.distbns[j].get(key, 0)
                    if pi > pj:
                        S.add(key)
                        vi += pi
                        vj += pj
        for x in sample:
            if not self.sparse and S[x]:
                countS += 1
            elif self.sparse and x in S:
                countS += 1        
        vsamp = countS / len(sample)
        if abs(vi - vsamp) < abs(vj - vsamp):
            return True
        else:
            return False

    def runAllPairs(self, sample, distbnIDs=None):
        if distbnIDs is None:
            distbnIDs = np.arange(self.K, dtype=int)
        K = len(distbnIDs)
        wins = np.zeros(K)
        for i in range(K):
            for j in range(i):
                if self.scheffeTest(distbnIDs[i], distbnIDs[j], sample):
                    wins[i] += 1
                else:
                    wins[j] += 1
        winner = distbnIDs[np.argmax(wins)]
        nOps = K * (K - 1) / 2 * len(sample)
        return winner, nOps

    def runTournament(self, sample, fast=False, fastParam=5, nAllPairs=None):
        assert(math.log(self.K, 2) == int(math.log(self.K, 2))) # K is a power of 2
        if nAllPairs is None:
            print('Using default nAllPairs')
            nAllPairs = int(math.pow(self.K, 1/3))
        survivors = np.array([x for x in range(self.K)], dtype=int)
        allPairs = set()
        nOps = 0
        np.random.shuffle(survivors)
        round = 1
        while len(survivors) > max(nAllPairs, 1):
            if nAllPairs > 0:
                allPairs = allPairs.union(set(np.random.choice(survivors, nAllPairs, replace=False).astype(int))) # sample some distbns for end
            newSurvivors = np.zeros(len(survivors)//2, dtype=int) # array to store winners
            nSamples = len(sample)
            subsample = sample
            if fast and fastParam*round < len(sample): # Fast tournament using fewer samples in early rounds
                nSamples = fastParam * round
                subsample = np.random.choice(sample, nSamples, replace=False)
            for i in range(0, len(survivors), 2):
                nOps += nSamples
                if self.scheffeTest(survivors[i], survivors[i+1], subsample): # left wins
                    newSurvivors[i//2] = survivors[i]
                else: # right wins
                    newSurvivors[i//2] = survivors[i+1]
            survivors = newSurvivors
            round += 1
        if len(allPairs) > 0:
            allPairs = allPairs.union(survivors)
            winner, nOpsAllPairs = self.runAllPairs(sample, list(allPairs))
        else:
            assert(len(survivors) == 1)
            winner = survivors[0]
            nOpsAllPairs = 0
        return winner, nOps + nOpsAllPairs

    def L1NN(self, sample):
        '''Return the L1 nearest neighbor'''
        pass

def generateHalfUnif(K, N):
    '''Generate distributions which are uniform over a random half of the domain'''
    distbns = np.zeros((K, N), dtype=float)
    for k in range(K):
        distbns[k, np.random.choice(N, N//2, replace=False)] = 1/(N//2)
    return distbns

def generateZipf(K, N, alpha):
    distbns = np.zeros((K,N), dtype=float)
    probs = 1 / np.arange(1, N+1, dtype=float)**alpha
    probs = probs / np.sum(probs)
    for k in range(K):
        np.random.shuffle(probs)
        distbns[k,:] = probs
    return distbns

def test1(trials=100, verbose=False):
    # np.random.seed(15613)
    K = 256
    N = 100
    nSamples = 40
    fastParam = 10
    acc = [0, 0, 0]
    ops = [0, 0, 0]
    print('K {}, N {}, S {}'.format(K, N, nSamples))
    print('FastTournament constant: {}'.format(fastParam))
    distbns = DistbnNN(K, N)
    distbns.setDistbns(generateHalfUnif(K, N))
    trueID = np.random.choice(K)
    sample = distbns.sampleDistbn(trueID, nSamples)
    print('Starting trials')
    for t in range(trials):
        # All Pairs
        # winner, nOps = distbns.runAllPairs(sample)
        # if winner == trueID:
        #     acc[0] += 1
        # ops[0] += nOps
        # if verbose:
        #     print('AllPairs:       TrueID {} Winner {} nOps {}'.format(trueID, winner, nOps))
        # Slow Tournament
        winner, nOps = distbns.runTournament(sample, fast=False)
        if winner == trueID:
            acc[1] += 1
        ops[1] += nOps
        if verbose:
            print('SlowTournament: TrueID {} Winner {} nOps {}'.format(trueID, winner, nOps))
        # Fast Tournament
        winner, nOps = distbns.runTournament(sample, fast=True, fastParam=fastParam)
        if winner == trueID:
            acc[2] += 1
        ops[2] += nOps
        if verbose:
            print('FastTournament: TrueID {} Winner {} nOps {}'.format(trueID, winner, nOps))
    print('AllPairs:       Accuracy {}, AvgOps {}'.format(acc[0]/trials, ops[0]/trials))
    print('SlowTournament: Accuracy {}, AvgOps {}'.format(acc[1]/trials, ops[1]/trials))
    print('FastTournament: Accuracy {}, AvgOps {}'.format(acc[2]/trials, ops[2]/trials))

def test2(): # testing sparse
    pass

if __name__ == '__main__':
    # test1(20, False)
    print(generateZipf(20, 100, 1))