import distbnnn
import numpy as np

def gridSearchTournament(fixedParams, nSamples, fastConst, nAllPairs, halfUnif):
    K = fixedParams['K']
    N = fixedParams['N']
    Q = fixedParams['Q']
    T = fixedParams['T']
    print('K {}, N {}, Q {}, T {}'.format(K, N, Q, T))
    distbns = distbnnn.DistbnNN(K, N)
    if halfUnif:
        distbns.setDistbns(distbnnn.generateHalfUnif(K, N))
    else:
        distbns.setDistbns(distbnnn.generateZipf(K, N, 2))
    testIDs = np.random.choice(K, Q, replace=False)
    for S in nSamples:
        for C in fastConst:
            for L in nAllPairs:
                print('S {}, C {}, L {}'.format(S, C, L))
                acc = [0, 0]
                ops = [0, 0]
                for q in range(Q):
                    trueID = testIDs[q]
                    for t in range(T):
                        sample = distbns.sampleDistbn(trueID, S)
                        winner, nOps = distbns.runTournament(sample, fast=False, nAllPairs=L)
                        if winner == trueID:
                            acc[0] += 1
                        ops[0] += nOps
                        winner, nOps = distbns.runTournament(sample, fast=True, fastParam=C, nAllPairs=L)
                        if winner == trueID:
                            acc[1] += 1
                        ops[1] += nOps
                print('SlowTournament: Accuracy {}, AvgOps {}'.format(acc[0]/(Q*T), ops[0]/(Q*T)))
                print('FastTournament: Accuracy {}, AvgOps {}'.format(acc[1]/(Q*T), ops[1]/(Q*T)))
                print()

if __name__ == '__main__':
    # np.random.seed(1832)
    # fixedParams = {
    #     'K': 8192,
    #     'N': 500,
    #     'Q': 20,
    #     'T': 5
    # }
    fixedParams = {
        'K': 4096,
        'N': 250,
        'Q': 20,
        'T': 5
    }
    # nSamples = [30, 40, 50, 60]
    # nSamples = [30, 40]
    nSamples = [20, 30, 40]
    fastConst = [5, 10, 15]
    nAllPairs = [0, 5, 10, 15, 20]
    # fastConst = [5, 10, 15, 20]
    # nAllPairs = [0, 10, 20, 30]
    gridSearchTournament(fixedParams, nSamples, fastConst, nAllPairs, halfUnif=False)
