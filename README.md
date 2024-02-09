- [(I)PSID: (Input) Preferential subspace identification  \[Python implementation\]](#ipsid-input-preferential-subspace-identification--python-implementation)
- [Publications](#publications)
  - [PSID](#psid)
  - [IPSID](#ipsid)
- [Usage guide](#usage-guide)
  - [Installation](#installation)
  - [Initialization](#initialization)
  - [Main learning function](#main-learning-function)
  - [Extracting latent states using learned model](#extracting-latent-states-using-learned-model)
  - [Required preprocessing](#required-preprocessing)
  - [Choosing the hyperparameters](#choosing-the-hyperparameters)
    - [How to pick the state dimensions nx and n1?](#how-to-pick-the-state-dimensions-nx-and-n1)
    - [How to pick the horizon `i`?](#how-to-pick-the-horizon-i)
- [Usage examples](#usage-examples)
  - [PSID](#psid-1)
  - [IPSID](#ipsid-1)
- [Change Log](#change-log)
- [Licence](#licence)

# (I)PSID: (Input) Preferential subspace identification <br/> [Python implementation]

For MATLAB implementation see http://github.com/ShanechiLab/PSID

Given signals y_t (e.g. neural signals) and z_t (e.g behavior), PSID learns a dynamic model for y_t while prioritizing the dynamics that are relevant to z_t. 

IPSID is an extension of PSID that also supports taking a third signal u_t (e.g., task instructions) that is simultaneously measured with y_t. In the learned dynamical model, u_t plays the role of input to the latent states. 

# Publications
## PSID
For the derivation of PSID and results in real neural data see the paper below.

Omid G. Sani, Hamidreza Abbaspourazad, Yan T. Wong, Bijan Pesaran, Maryam M. Shanechi. *Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification*. Nature Neuroscience, 24, 140–149 (2021). https://doi.org/10.1038/s41593-020-00733-0

View-only full-text link: https://rdcu.be/b993t

Original preprint: https://doi.org/10.1101/808154

You can also find a summary of the paper in the following Twitter thread:
https://twitter.com/MaryamShanechi/status/1325835609345122304

## IPSID 
For the derivation of IPSID and results in real neural data see the paper below.

Parsa Vahidi*, Omid G. Sani*, Maryam M. Shanechi. *Modeling and dissociation of intrinsic and input-driven neural population dynamics underlying behavior*. PNAS (2024). https://doi.org/10.1073/pnas.2212887121


# Usage guide
## Installation
Download the source code from [the GitHub repository](https://github.com/ShanechiLab/PyPSID), or install PSID in your Python environment using pip, by running:
```
pip install PSID --upgrade
```
You can find the usage license in [LICENSE.md](https://github.com/ShanechiLab/PyPSID/blob/main/LICENSE.md).

## Initialization
Import the PSID module.
```
import PSID
```

## Main learning function
The main functions for the Python implementation are the follwing:
- For PSID: [source/PSID/PSID.py](https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/PSID.py) -> the function called PSID
- For IPSID [source/PSID/IPSID.py](https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/IPSID.py) -> the function called IPSID

A complete usage guide is available in as comments in each function. The following shows example use cases:
```
idSys = PSID.PSID(y, z, nx, n1, i)
# Or, if modeling effect of input u is also of interest
idSys = PSID.IPSID(y, z, u, nx, n1, i)
```
Inputs:
- y and z are time x dimension matrices with neural (e.g. LFP signal powers or spike counts) and behavioral data (e.g. joint angles, hand position, etc), respectively. 
- IPSID also takes u as an input, which is a time x dimension matrix, containing the measured input data. 
- nx is the total number of latent states to be identified.
- n1 is the number of states that are going to be dedicated to behaviorally relevant dynamics.
- i is the subspace horizon used for modeling. 

Output:
- idSys: an LSSM object containing all model parameters (A, Cy, Cz, etc). For a full list see the code.

## Extracting latent states using learned model
Once a model is learned using (I)PSID, you can apply the model to new data (i.e. run the associated Kalman filter) as follows:
```
zPred, yPred, xPred = idSys.predict(y)
# Or, for IPSID:
zPred, yPred, xPred = idSys.predict(y, u)
```
Input:
- y: neural activity time series (time x dimension)
- [For IPSID] u: input time series (time x dimension)

Outputs:
- zPred: one-step ahead prediction of behavior (if any)
- yPred: one-step ahead prediction of neural activity
- xPred: Extracted latent state

## Required preprocessing
- Repeated data dimensions (e.g., two identical neurons) can cause issues for the learning. Remove repeated data dimensions as a preprocessing and repeat predictions as needed to reproduce prediction of repeated data dimensions. 
- A required preprocessing when using (I)PSID is to remove the mean of neural/behavior/input signals and if needed, add them back to neural/behavior predictions after learning the model. Starting from version 1.1.0, Python (I)PSID and MATLAB PSID libraries automatically do this by default so that users won't need to worry about it. Please update to the latest version if you are using an older version.

## Choosing the hyperparameters
### How to pick the state dimensions nx and n1?
nx determines the total dimension of the latent state and n1 determines how many of those dimensions will be prioritizing the inclusion of behaviorally relevant neural dynamics (i.e. will be extracted using stage 1 of (I)PSID). So the values that you would select for these hyperparameters depend on the goal of modeling and on the data. Some examples use cases are:

If you want to perform dimension reduction, nx will be your desired target dimension. For example, to reduce dimension to 2 to plot low-dimensional visualizations of neural activity, you would use nx=2. Now if you want to reduce dimension while preserving as much behaviorally relevant neural dynamics as possible, you would use n1=nx.
If you want to find the best fit to data overall, you can perform a grid search over values of nx and n1 and pick the value that achieves the best performance metric in the training data. For example, you could pick the nx and n1 pair that achieves the best cross-validated behavior decoding in an inner-cross-validation within the training data.

### How to pick the horizon `i`?
The horizon `i` does not affect the model structure and only affects the intermediate linear algebra operations that (I)PSID performs during the learning of the model. Nevertheless, different values of `i` may have different model learning performance. `i` needs to be at least 2, but also also determines the maximum n1 and nx that can be used per:

```
n1 <= nz * i
nx <= ny * i
```

So if you have a low dimensional y_k or z_k (small ny or nz), you typically would choose larger values for `i`, and vice versa. It is also possible to select the best performing `i` via an inner cross-validation approach similar to nx and n1 above. Overall, since `i` affects the learning performance, it is important for reproducibility that the `i` that was used is reported.

For more information, see the notebook(s) referenced in the next section. 

# Usage examples
## PSID
Example code for running PSID is provided in 
[source/example/PSID_example.py](https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/PSID_example.py)
This script performs PSID model identification and visualizes the learned eigenvalues similar to in Supplementary Fig 1 in (Sani et al, 2021).

The following notebook also contains some examples along with more descriptions:
[source/example/PSID_tutorial.ipynb](https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/PSID_tutorial.ipynb)

## IPSID
Example code for running IPSID is provided in 
[source/example/IPSID_example.py](https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/IPSID_example.py)
This script performs IPSID model identification and visualizes the learned eigenvalues similar to in Fig. 2A in (Vahidi, Sani, et al, 2024).

The following notebook also contains some examples along with more descriptions:
[source/example/IPSID_tutorial.ipynb](https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/IPSID_tutorial.ipynb)

# Change Log
You can see the change log in in [ChangeLog.md](https://github.com/ShanechiLab/PyPSID/blob/main/ChangeLog.md)  

# Licence
Copyright (c) 2020 University of Southern California  
See full notice in [LICENSE.md](https://github.com/ShanechiLab/PyPSID/blob/main/LICENSE.md)  
Omid G. Sani, Parsa Vahidi and Maryam M. Shanechi  
Shanechi Lab, University of Southern California
