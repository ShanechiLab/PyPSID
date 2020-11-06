""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Example for using the PSID algorithm
"""

import argparse, io, os, copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import PSID
from PSID.evaluation import evalPrediction
from PSID.MatHelper import loadmat

def main():
    parser = argparse.ArgumentParser(description='Run PSID on an example simulated dataset')
    parser.add_argument('--datafile', type=str, default='./source/example/sample_data.mat', help='Data file')
    
    args = parser.parse_args()

    # Load data                    
    data = loadmat(args.datafile)
    # This data is generated from a system (shown in Supplementary Fig. 1) with 
    # (a) 2 behaviorally relevant latent states, 
    # (b) 2 behaviorally irrelevant latent states, and 
    # (c) 2 states that drive behavior but are not represented in neural activity

    allYData = data['y']
    allZData = data['z']
    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5*allYData.shape[0]), dtype=int)
    testInds = np.arange(1+trainInds[-1], allYData.shape[0])
    yTrain = allYData[trainInds, :]
    yTest = allYData[testInds, :]
    zTrain = allZData[trainInds, :]
    zTest = allZData[testInds, :]

    ## (Example 1) PSID can be used to dissociate and extract only the 
    # behaviorally relevant latent states (with nx = n1 = 2)
    idSys1 = PSID.PSID(yTrain, zTrain, nx=2, n1=2, i=10)
    # You can also use the time_first=False argument if time is the second dimension:
    # idSys1 = PSID.PSID(yTrain.T, zTrain.T, nx=2, n1=2, i=10, time_first=False) 

    # Predict behavior using the learned model
    zTestPred1, yTestPred1, xTestPred1 = idSys1.predict(yTest)

    # Compute CC of decoding
    nz = zTest.shape[1]
    CC = evalPrediction(zTest, zTestPred1, 'CC')

    # Predict behavior using the true model for comparison
    trueSys = PSID.LSSM(params=data['trueSys'])
    if not hasattr(trueSys, 'Cz'):
      trueSys.Cz = trueSys.T[1:, :].T
    zTestPredIdeal, yTestPredIdeal, xTestPredIdeal = trueSys.predict(yTest)
    CCIdeal = evalPrediction(zTest, zTestPredIdeal, 'CC')

    print('PSID decoding CC = {:.3g}, ideal decoding CC using true model = {:.3g}'.format(np.mean(CC), np.mean(CCIdeal)) )
    
    ## (Example 2) Optionally, PSID can additionally also learn the 
    # behaviorally irrelevant latent states (with nx = 4, n1 = 2)
    idSys2 = PSID.PSID(yTrain, zTrain, nx=4, n1=2, i=10)

    ## (Example 3) PSID can be used if data is available in discontinuous segments (e.g. different trials)
    # In this case, y and z data segments must be provided as elements of a list
    # Trials do not need to have the same number of samples
    # Here, for example assume that trials start at every 1000 samples.
    # And each each trial has a random length of 500 to 900 samples
    trialStartInds = np.arange(0, allYData.shape[0]-1000, 1000)
    trialDurRange = np.array([900, 990])
    trialDur = np.random.randint(low=trialDurRange[0], high=1+trialDurRange[1], size=trialStartInds.shape)
    trialInds = [trialStartInds[ti]+np.arange(trialDur[ti]) for ti in range(trialStartInds.size)] 
    yTrials = [allYData[trialIndsThis, :] for trialIndsThis in trialInds] 
    zTrials = [allZData[trialIndsThis, :] for trialIndsThis in trialInds] 

    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5*len(yTrials)), dtype=int)
    testInds = np.arange(1+trainInds[-1], len(yTrials))
    yTrain = [yTrials[ti] for ti in trainInds]
    yTest = [yTrials[ti] for ti in testInds]
    zTrain = [zTrials[ti] for ti in trainInds]
    zTest = [zTrials[ti] for ti in testInds]

    idSys3 = PSID.PSID(yTrain, zTrain, nx=2, n1=2, i=10)

    for ti in range(len(yTest)):
      zPredThis, yPredThis, xPredThis = idSys3.predict(yTest[ti])
      if ti == 0:
        zTestA = zTest[ti]
        zPredA = zPredThis
      else:
        zTestA = np.concatenate( (zTestA, zTest[ti]), axis=0)
        zPredA = np.concatenate( (zPredA, zPredThis), axis=0)

    CCTrialBased = evalPrediction(zTestA, zPredA, 'CC')

    print('PSID trial based decoding CC = {:.3g}, ideal decoding CC using true model = {:.3g}'.format(np.mean(CCTrialBased), np.mean(CCIdeal)) )

    # #########################################
    # Plot the true and identified eigenvalues    

    # (Example 1) Eigenvalues when only learning behaviorally relevant states
    idEigs1 = np.linalg.eig(idSys1.A)[0]

    # (Example 2) Additional eigenvalues when also learning behaviorally irrelevant states
    # The identified model is already in form of Eq. 4, with behaviorally irrelevant states 
    # coming as the last 2 dimensions of the states in the identified model
    idEigs2 = np.linalg.eig(idSys2.A[2:, 2:])[0]

    relevantDims = trueSys.zDims - 1 # Dimensions that drive both behavior and neural activity
    irrelevantDims = [x for x in np.arange(trueSys.state_dim, dtype=int) if x not in relevantDims] # Dimensions that only drive the neural activity
    trueEigsRelevant = np.linalg.eig(trueSys.A[np.ix_(relevantDims, relevantDims)])[0]
    trueEigsIrrelevant = np.linalg.eig(trueSys.A[np.ix_(irrelevantDims, irrelevantDims)])[0]
    nonEncodedEigs = np.linalg.eig(data['epsSys']['a'])[0] # Eigenvalues for states that only drive behavior

    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2)
    axs[1].remove() 
    ax = axs[0]
    ax.axis('equal')
    ax.add_patch( patches.Circle((0,0), radius=1, fill=False, color='black', alpha=0.2, ls='-') )
    ax.plot([-1,1,0,0,0], [0,0,0,-1,1], color='black', alpha=0.2, ls='-')
    ax.scatter(np.real(nonEncodedEigs), np.imag(nonEncodedEigs), marker='o', edgecolors='#0000ff', facecolors='none', label='Not encoded in neural signals')
    ax.scatter(np.real(trueEigsIrrelevant), np.imag(trueEigsIrrelevant), marker='o', edgecolors='#ff0000', facecolors='none', label='Behaviorally irrelevant')
    ax.scatter(np.real(trueEigsRelevant), np.imag(trueEigsRelevant), marker='o', edgecolors='#00ff00', facecolors='none', label='Behaviorally relevant')
    ax.scatter(np.real(idEigs1), np.imag(idEigs1), marker='x', facecolors='#00aa00', label='PSID Identified (stage 1)')
    ax.scatter(np.real(idEigs2), np.imag(idEigs2), marker='x', facecolors='#aa0000', label='(optional) PSID Identified (stage 2)')
    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.show()

    pass


if __name__ == '__main__':
  main()