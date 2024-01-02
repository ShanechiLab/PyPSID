""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Parsa Vahidi, Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Example for using the IPSID algorithm
"""

import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import PSID
from PSID.evaluation import evalPrediction
from PSID.MatHelper import loadmat
from PSID.PrepModel import PrepModel

def main():
    
    # (1) IPSID examples
    #################################
    #################################
    sample_model_path = os.path.join(os.path.dirname(PSID.__file__), 'example', 'sample_model_IPSID.mat')
    
    parser = argparse.ArgumentParser(description='Run IPSID on an example simulated dataset')
    parser.add_argument('--datafile', type=str, default=sample_model_path, help='Data file')
    
    args = parser.parse_args()

    # Load data
    print('Loading example model from {}'.format(args.datafile))
    data = loadmat(args.datafile)
    # This is an example model (shown in Fig. 2A) with 
    # (a) 2 behaviorally relevant latent states x_k^(1) (corresponding to intrinsic behaviorally relevant dynamics),
    # (b) 4 other latent states x_k^(2) (corresponding to other intrinsic dynamics),
    # (c) 2 states that drive the external input (corresponding to input dynamics)

    # Generating some sample data from this model
    np.random.seed(42) # For exact reproducibility

    N = int(2e5)

    # Generating dynamical input u
    uSys = PSID.LSSM(params=data['uSys'])
    u, _ = uSys.generateRealization(N)
    Au = uSys.A
    trueSys = PSID.LSSM(params=data['trueSys'])
    trueSys.Dz = trueSys.dz
    y, x = trueSys.generateRealization(N, u=u)
    z = (trueSys.Cz @ x.T).T + (trueSys.Dz @ u.T).T 

    # Add some z dynamics that are not encoded in y (i.e. epsilon)
    epsSys = PSID.LSSM(params=data['epsSys'])
    _, x_eps = epsSys.generateRealization(N)
    z += (epsSys.Cz @ x_eps.T).T

    allYData, allZData, allUData = y, z, u


    # Given the above state-space model used by IPSID, it is important for the neural/behavior/input data to be zero-mean. 
    # Starting version v1.1.0, IPSID by default internally removes the mean from the neural/behavior data and adds
    # it back to predictions, so the user does not need to handle this preprocessing. If the data is already zero-mean,
    # this mean-removal will simply subtract and add zeros to signals so everything will still work.
    # To cover this general case with data that is not zero-mean, let's artificially add some non-zero mean to the sample data:
    YMean = 10*np.random.randn(allYData.shape[-1])
    ZMean = 10*np.random.randn(allZData.shape[-1])
    UMean = 10*np.random.randn(allUData.shape[-1])
    allYData += YMean
    allZData += ZMean
    allUData += UMean
    # Also reflect this in the true model:
    trueSys.YPrepModel = PrepModel(mean=YMean, remove_mean=True)
    trueSys.ZPrepModel = PrepModel(mean=ZMean, remove_mean=True)
    trueSys.UPrepModel = PrepModel(mean=UMean, remove_mean=True)
    

    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5*allYData.shape[0]), dtype=int)
    testInds = np.arange(1+trainInds[-1], allYData.shape[0])
    yTrain = allYData[trainInds, :]
    yTest = allYData[testInds, :]
    zTrain = allZData[trainInds, :]
    zTest = allZData[testInds, :]
    uTrain = allUData[trainInds, :]
    uTest = allUData[testInds, :]
    
    ## (Example 1) IPSID can be used to dissociate and extract only the 
    # intrinsic behaviorally relevant latent states (with nx = n1 = 2)
    idSys1 = PSID.IPSID(yTrain, zTrain, uTrain, nx=2, n1=2, i=10)
    # You can also use the time_first=False argument if time is the second dimension:
    # idSys1 = PSID.IPSID(yTrain.T, zTrain.T, uTrain.T, nx=2, n1=2, i=10, time_first=False) 
    
    # Predict behavior using the learned model
    zTestPred1, yTestPred1, xTestPred1 = idSys1.predict(yTest, uTest)

    # Compute R2 of decoding
    R2 = evalPrediction(zTest, zTestPred1, 'R2')

    # Predict behavior using the true model for comparison
    zTestPredIdeal, yTestPredIdeal, xTestPredIdeal = trueSys.predict(yTest, uTest)
    R2Ideal = evalPrediction(zTest, zTestPredIdeal, 'R2')

    print('Behavior decoding R2:\n  IPSID => {:.3g}, Ideal using true model => {:.3g}'.format(np.mean(R2), np.mean(R2Ideal)) )
    
    ## (Example 2) Optionally, IPSID can additionally also learn the 
    # behaviorally irrelevant latent states (with nx = 6, n1 = 2)
    idSys2 = PSID.IPSID(yTrain, zTrain, uTrain, nx=6, n1=2, i=10)

    # In addition to ideal behavior decoding, this model will also have ideal neural self-prediction 
    zTestPred2, yTestPred2, xTestPred2 = idSys2.predict(yTest, uTest)
    yR2 = evalPrediction(yTest, yTestPred2, 'R2')
    yR2Ideal = evalPrediction(yTest, yTestPredIdeal, 'R2')
    print('Neural self-prediction R2:\n  IPSID => {:.3g}, Ideal using true model => {:.3g}'.format(np.mean(yR2), np.mean(yR2Ideal)))

    ## (Example 3) IPSID can be used if data is available in discontinuous segments (e.g. different trials)
    # In this case, y, z and u data segments must be provided as elements of a list
    # Trials do not need to have the same number of samples
    # Here, for example assume that trials start at every 1000 samples.
    # And each each trial has a random length of 900 to 990 samples
    trialStartInds = np.arange(0, allYData.shape[0]-1000, 1000)
    trialDurRange = np.array([900, 990])
    trialDur = np.random.randint(low=trialDurRange[0], high=1+trialDurRange[1], size=trialStartInds.shape)
    trialInds = [trialStartInds[ti]+np.arange(trialDur[ti]) for ti in range(trialStartInds.size)] 
    yTrials = [allYData[trialIndsThis, :] for trialIndsThis in trialInds] 
    zTrials = [allZData[trialIndsThis, :] for trialIndsThis in trialInds] 
    uTrials = [allUData[trialIndsThis, :] for trialIndsThis in trialInds] 

    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5*len(yTrials)), dtype=int)
    testInds = np.arange(1+trainInds[-1], len(yTrials))
    yTrainTrials = [yTrials[ti] for ti in trainInds]
    yTestTrials = [yTrials[ti] for ti in testInds]
    zTrainTrials = [zTrials[ti] for ti in trainInds]
    zTestTrials = [zTrials[ti] for ti in testInds]
    uTrainTrials = [uTrials[ti] for ti in trainInds]
    uTestTrials = [uTrials[ti] for ti in testInds]

    idSys3 = PSID.IPSID(yTrainTrials, zTrainTrials, uTrainTrials, nx=2, n1=2, i=10)

    zPredTrials, yPredTrials, xPredTrials = idSys3.predict(yTestTrials, uTestTrials)
    zPredTrialsIdeal, yPredTrialsIdeal, xPredTrialsIdeal = trueSys.predict(yTestTrials, uTestTrials)
    zTestA = np.concatenate( zTestTrials, axis=0)
    zPredA = np.concatenate( zPredTrials, axis=0)
    zPredIdealA = np.concatenate( zPredTrialsIdeal, axis=0)

    R2TrialBased = evalPrediction(zTestA, zPredA, 'R2')
    R2TrialBasedIdeal = evalPrediction(zTestA, zPredIdealA, 'R2')

    print('Behavior decoding R2 (trial-based learning/decoding):\n  IPSID => {:.3g}, Ideal using true model = {:.3g}'.format(np.mean(R2TrialBased), np.mean(R2TrialBasedIdeal)) )

    # #########################################
    # Plot the true and identified eigenvalues    

    # (Example 1) Eigenvalues when only learning behaviorally relevant states
    idEigs1 = np.linalg.eig(idSys1.A)[0]

    # (Example 2) Additional eigenvalues when also learning behaviorally irrelevant states
    # The identified model is already in form of Eq. 1, with behaviorally irrelevant states 
    # coming as the last 4 dimensions of the states in the identified model
    idEigs2 = np.linalg.eig(idSys2.A[2:, 2:])[0]

    relevantDims = trueSys.zDims - 1 # Dimensions that drive both behavior and neural activity
    irrelevantDims = [x for x in np.arange(trueSys.state_dim, dtype=int) if x not in relevantDims] # Dimensions that only drive the neural activity
    trueEigsRelevant = np.linalg.eig(trueSys.A[np.ix_(relevantDims, relevantDims)])[0]
    trueEigsIrrelevant = np.linalg.eig(trueSys.A[np.ix_(irrelevantDims, irrelevantDims)])[0]
    trueEigsInput = np.linalg.eig(Au)[0]

    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2)
    axs[1].remove() 
    ax = axs[0]
    ax.axis('equal')
    ax.add_patch( patches.Circle((0,0), radius=1, fill=False, color='black', alpha=0.2, ls='-') )
    ax.plot([-1,1,0,0,0], [0,0,0,-1,1], color='black', alpha=0.2, ls='-')
    ax.scatter(np.real(trueEigsInput), np.imag(trueEigsInput), marker='o', edgecolors='#800080', facecolors='none', label='Input eigenvalues')
    ax.scatter(np.real(trueEigsIrrelevant), np.imag(trueEigsIrrelevant), marker='o', edgecolors='#FF5733', facecolors='none', label='Other neural eigenvalues')
    ax.scatter(np.real(trueEigsRelevant), np.imag(trueEigsRelevant), marker='o', edgecolors='#50C878', facecolors='none', label='Behaviorally relevant neural eigenvalues')
    ax.scatter(np.real(idEigs1), np.imag(idEigs1), marker='x', facecolors='#138a33', label='IPSID Identified (stage 1)')
    ax.scatter(np.real(idEigs2), np.imag(idEigs2), marker='x', facecolors='#b04c1a', label='(optional) IPSID Identified (stage 2)')
    ax.set_title('True and identified eigevalues')
    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.show()



    # (2) IPSID example with the additional steps
    ###################################################
    ###################################################
    sample_model_path = os.path.join(os.path.dirname(PSID.__file__), 'example', 'sample_model_IPSID_add_step.mat')
    
    parser = argparse.ArgumentParser(description='Run IPSID with the additional steps on an example simulated dataset')
    parser.add_argument('--datafile', type=str, default=sample_model_path, help='Data file')
    
    args = parser.parse_args()

    # Load data
    print('Loading example model from {}'.format(args.datafile))
    data = loadmat(args.datafile)
    # This is an example model (shown in Fig. 3) with 
    # (a) 2 behaviorally relevant latent states, x_k^(1), encoded in neural activity y_k (corresponding to intrinsic behaviorally relevant neural dynamics), 
    # (b) 2 other latent states, x_k^(2), encoded in neural activity y_k (corresponding to other intrinsic dynamics),
    # (c) 2 states that drive the external input (corresponding to input dynamics)
    # (d) 2 behaviorally relevant latent states, x_k^(3), driven by the input u_k but not encoded in neural activity y_k
    
    # Generating some sample data from this model
    np.random.seed(42) # For exact reproducibility

    N = int(2e5)

    # Generating dynamical input u
    uSys = PSID.LSSM(params=data['uSys'])
    u, _ = uSys.generateRealization(N)
    Au = uSys.A
    trueSys = PSID.LSSM(params=data['trueSys'])
    y, x = trueSys.generateRealization(N, u=u)
    z = (trueSys.Cz @ x.T).T 

    allYData, allZData, allUData = y, z, u

    # Given the above state-space model used by IPSID, it is important for the neural/behavior/input data to be zero-mean. 
    # Starting version v1.1.0, IPSID by default internally removes the mean from the neural/behavior data and adds
    # it back to predictions, so the user does not need to handle this preprocessing. If the data is already zero-mean,
    # this mean-removal will simply subtract and add zeros to signals so everything will still work.
    # To cover this general case with data that is not zero-mean, let's artificially add some non-zero mean to the sample data:
    YMean = 10*np.random.randn(allYData.shape[-1])
    ZMean = 10*np.random.randn(allZData.shape[-1])
    UMean = 10*np.random.randn(allUData.shape[-1])
    allYData += YMean
    allZData += ZMean
    allUData += UMean
    # Also reflect this in the true model:
    trueSys.YPrepModel = PrepModel(mean=YMean, remove_mean=True)
    trueSys.ZPrepModel = PrepModel(mean=ZMean, remove_mean=True)
    trueSys.UPrepModel = PrepModel(mean=UMean, remove_mean=True)
    

    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5*allYData.shape[0]), dtype=int)
    testInds = np.arange(1+trainInds[-1], allYData.shape[0])
    yTrain = allYData[trainInds, :]
    yTest = allYData[testInds, :]
    zTrain = allZData[trainInds, :]
    zTest = allZData[testInds, :]
    uTrain = allUData[trainInds, :]
    uTest = allUData[testInds, :]


    
    ## (Example 3) IPSID with additional steps can be used to further 
    #  dissociate the intrinsic behaviorally relevant neural dynamics that 
    #  encoded in neural activity from those that are not.
    
    # all latent states [x1;x2,x3] (with nx = 6, n1 = 2, n3 = 2)
    idSys4 = PSID.IPSID(yTrain, zTrain, uTrain, nx=6, n1=2, i=10, remove_nonYrelated_fromX1=True, n_pre=4, n3=2) # n_pre should be equal to true n1+true n2

    # Predict behavior using the learned model
    zTestPred4, yTestPred4, xTestPred3 = idSys4.predict(yTest, uTest)

    # Compute R2 of decoding and neural self-prediction
    R2 = evalPrediction(zTest, zTestPred4, 'R2')
    yR2 = evalPrediction(yTest, yTestPred4, 'R2')
    
    # For comparison, let's also learn a model without the additional step 2 (only [x1;x2])
    idSys4_low_dim = PSID.IPSID(yTrain, zTrain, uTrain, nx=4, n1=2, i=10, remove_nonYrelated_fromX1=True, n_pre=4, n3=0) # n_pre should be equal to true n1+true n2
    zTestPred4_low_dim, yTestPred4_low_dim, xTestPred4_low_dim = idSys4_low_dim.predict(yTest, uTest)
    R2_low_dim = evalPrediction(zTest, zTestPred4_low_dim, 'R2')
    yR2_low_dim = evalPrediction(yTest, yTestPred4_low_dim, 'R2')
    
    # Predict using the true model for comparison
    zTestPredIdeal, yTestPredIdeal, xTestPredIdeal = trueSys.predict(yTest, uTest)
    R2Ideal = evalPrediction(zTest, zTestPredIdeal, 'R2')
    yR2Ideal = evalPrediction(yTest, yTestPredIdeal, 'R2')

    print('Behavior decoding R2:\n  IPSID => {:.3g}, IPSID (without additional step 2) => {:.3g}, Ideal using true model => {:.3g}'.format(np.mean(R2), np.mean(R2_low_dim), np.mean(R2Ideal)) )
    print('Neural self-prediction R2:\n  IPSID => {:.3g}, IPSID (without additional step 2) => {:.3g}, Ideal using true model => {:.3g}'.format(np.mean(yR2), np.mean(yR2_low_dim), np.mean(yR2Ideal)) )

    # #########################################
    # Plot the true and identified eigenvalues for IPSID with additional steps   

    # Intrinsic behaviorally relevant eigenvalues encoded in neural activity
    idEigs1 = np.linalg.eig(idSys4_low_dim.A[:2,:2])[0]

    # Other intrinsic eigenvalues encoded in neural activity
    idEigs2 = np.linalg.eig(idSys4_low_dim.A[2:, 2:])[0]

    # Behaviorally relevant eigenvalues not encoded in neural activity
    idEigs3 = np.linalg.eig(idSys4.A[4:, 4:])[0]
    
    relevantDims = trueSys.zDims - 1 # Dimensions that drive both behavior and neural activity
    irrelevantDims = [2, 3] # Dimensions that only drive the neural activity
    trueEigsRelevant = np.linalg.eig(trueSys.A[np.ix_(relevantDims, relevantDims)])[0]
    trueEigsIrrelevant = np.linalg.eig(trueSys.A[np.ix_(irrelevantDims, irrelevantDims)])[0]
    trueEigsInput = np.linalg.eig(Au)[0]
    trueEigsNonEncoded = np.linalg.eig(trueSys.A[4:,4:])[0]


    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2)
    axs[1].remove() 
    ax = axs[0]
    ax.axis('equal')
    ax.add_patch( patches.Circle((0,0), radius=1, fill=False, color='black', alpha=0.2, ls='-') )
    ax.plot([-1,1,0,0,0], [0,0,0,-1,1], color='black', alpha=0.2, ls='-')
    ax.scatter(np.real(trueEigsInput), np.imag(trueEigsInput), marker='o', edgecolors='#800080', facecolors='none', label='Input eigenvalues')
    ax.scatter(np.real(trueEigsIrrelevant), np.imag(trueEigsIrrelevant), marker='o', edgecolors='#FF5733', facecolors='none', label='Other neural eigenvalues')
    ax.scatter(np.real(trueEigsRelevant), np.imag(trueEigsRelevant), marker='o', edgecolors='#50C878', facecolors='none', label='Behaviorally relevant neural eigenvalues')
    ax.scatter(np.real(trueEigsNonEncoded), np.imag(trueEigsNonEncoded), marker='o', edgecolors='#000000', facecolors='none', label='Behaviorally relevant not encoded in neural activity eigenvalues')
    ax.scatter(np.real(idEigs1), np.imag(idEigs1), marker='x', facecolors='#138a33', label='IPSID Identified (stage 1)')
    ax.scatter(np.real(idEigs2), np.imag(idEigs2), marker='x', facecolors='#b04c1a', label='(optional) IPSID Identified (stage 2)')
    ax.scatter(np.real(idEigs3), np.imag(idEigs3), marker='x', facecolors='#000000', label='(optional) IPSID Identified in optional additional step 2')

    ax.set_title('True and identified eigevalues')
    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.show()

    ## (Example 4) IPSID with additional steps can also be used if data is available 
    # in discontinuous segments (e.g. different trials)
    # In this case, y, z and u data segments must be provided as elements of a list
    # Trials do not need to have the same number of samples
    # Here, for example assume that trials start at every 1000 samples.
    # And each each trial has a random length of 900 to 990 samples
    trialStartInds = np.arange(0, allYData.shape[0]-1000, 1000)
    trialDurRange = np.array([900, 990])
    trialDur = np.random.randint(low=trialDurRange[0], high=1+trialDurRange[1], size=trialStartInds.shape)
    trialInds = [trialStartInds[ti]+np.arange(trialDur[ti]) for ti in range(trialStartInds.size)] 
    yTrials = [allYData[trialIndsThis, :] for trialIndsThis in trialInds] 
    zTrials = [allZData[trialIndsThis, :] for trialIndsThis in trialInds] 
    uTrials = [allUData[trialIndsThis, :] for trialIndsThis in trialInds] 

    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5*len(yTrials)), dtype=int)
    testInds = np.arange(1+trainInds[-1], len(yTrials))
    yTrainTrials = [yTrials[ti] for ti in trainInds]
    yTestTrials = [yTrials[ti] for ti in testInds]
    zTrainTrials = [zTrials[ti] for ti in trainInds]
    zTestTrials = [zTrials[ti] for ti in testInds]
    uTrainTrials = [uTrials[ti] for ti in trainInds]
    uTestTrials = [uTrials[ti] for ti in testInds]

    idSys4 = PSID.IPSID(yTrainTrials, zTrainTrials, uTrainTrials, nx=6, n1=2, i=10, remove_nonYrelated_fromX1=True, n_pre=4, n3=2) # n_pre should be equal to true n1+true n2

    zPredTrials, yPredTrials, xPredTrials = idSys4.predict(yTestTrials, uTestTrials)
    zPredA = np.concatenate( zPredTrials, axis=0)
    yPredA = np.concatenate( yPredTrials, axis=0)

    zPredTrialsIdeal, yPredTrialsIdeal, xPredTrialsIdeal = trueSys.predict(yTestTrials, uTestTrials)
    zPredIdealA = np.concatenate( zPredTrialsIdeal, axis=0)
    yPredIdealA = np.concatenate( yPredTrialsIdeal, axis=0)

    zTestA = np.concatenate( zTestTrials, axis=0)
    yTestA = np.concatenate( yTestTrials, axis=0)
    R2TrialBased = evalPrediction(zTestA, zPredA, 'R2')
    yR2TrialBased = evalPrediction(yTestA, yPredA, 'R2')
    R2TrialBasedIdeal = evalPrediction(zTestA, zPredIdealA, 'R2')
    yR2TrialBasedIdeal = evalPrediction(yTestA, yPredIdealA, 'R2')

    # For comparison, let's also learn a model without the additional step 2 (only [x1;x2])
    idSys4_low_dim = PSID.IPSID(yTrainTrials, zTrainTrials, uTrainTrials, nx=4, n1=2, i=10, remove_nonYrelated_fromX1=True, n_pre=4, n3=0) # n_pre should be equal to true n1+true n2
    zPredTrials_low_dim, yPredTrials_low_dim, xPredTrials_low_dim = idSys4_low_dim.predict(yTestTrials, uTestTrials)
    zPredA_low_dim = np.concatenate( zPredTrials_low_dim, axis=0)
    yPredA_low_dim = np.concatenate( yPredTrials_low_dim, axis=0)
    R2TrialBased_low_dim = evalPrediction(zTestA, zPredA_low_dim, 'R2')
    yR2TrialBased_low_dim = evalPrediction(yTestA, yPredA_low_dim, 'R2')

    print('\nBehavior decoding R2 (trial-based learning/decoding):\n  IPSID => {:.3g}, IPSID (without additional step 2) => {:.3g}, Ideal using true model => {:.3g}'.format(np.mean(R2TrialBased), np.mean(R2TrialBased_low_dim), np.mean(R2TrialBasedIdeal)) )
    print('Neural self-prediction R2 (trial-based learning/decoding):\n  IPSID => {:.3g}, IPSID (without additional step 2) => {:.3g}, Ideal using true model => {:.3g}'.format(np.mean(yR2TrialBased), np.mean(yR2TrialBased_low_dim), np.mean(yR2TrialBasedIdeal)) )

    pass


if __name__ == '__main__':
  main()