# PSID: Preferential subspace identification <br/> [Python implementation]

For MATLAB implementation see [http://github.com/ShanechiLab/PSID](github.com/ShanechiLab/PSID)

Given signals y_t (e.g. neural signals) and z_t (e.g behavior), PSID learns a dynamic model for y_t while prioritizing the dynamics that are relevant to z_t. 

For the derivation and results in real neural data see:

Omid G. Sani, Bijan Pesaran, Maryam M. Shanechi  (2019). *Modeling behaviorally relevant neural dynamics using Preferential Subspace IDentification (PSID)* bioRxiv 808154, doi: 10.1101/808154, https://doi.org/10.1101/808154


# Usage guide
## Initialization
Import the PSID module.
```
import PSID
```

## Main learning function
The main function for the Python implementation is [source/PSID/PSID.py](source/PSID/PSID.py) -> function PSID. A complete usage guide is available in the function. The following shows an example case:
```
idSys = PSID.PSID(y, z, nx, n1, i);
```
Inputs:
- y and z are time x dimension matrices with neural (e.g. LFP signal powers or spike counts) and behavioral data (e.g. joint angles, hand position, etc), respectively. 
- nx is the total number of latent states to be identified.
- n1 is the number of states that are going to be dedicated to behaviorally relevant dynamics.
- i is the subspace horizon used for modeling. 

Output:
- idSys: an LSSM object containing all model parameters (A, Cy, Cz, etc). For a full list see the code.

## Extracting latent states using learned model
Once a model is learned using PSID, you can apply the model to new data (i.e. run the associated Kalman filter) as follows:
```
zPred, yPred, xPred = idSys.predict(y)
```
Input:
- y: neural activity time series (time x dimension)

Outputs:
- zPred: one-step ahead prediction of behavior (if any)
- yPred: one-step ahead prediction of neural activity
- xPred: Extracted latent state

# Example script
Example simulated data and the code for running PSID on the data is provided in 
[source/PSID_example.py](source/PSID_example.py)
This script performs PSID model identification and visualizes the learned eigenvalues similar to in Supplementary Fig 1.

# Licence
Copyright (c) 2020 University of Southern California  
See full notice in [LICENSE.md](LICENSE.md)  
Omid G. Sani and Maryam M. Shanechi  
Shanechi Lab, University of Southern California
