# Changes 
Versioning follows [semver](https://semver.org/).

- v1.2.6:
  - Fixes minor error in variable init for trial-based ISID.
- v1.2.5:
  - Fixes minor `numpy.eye` error that was thrown for unstable learned models.
- v1.2.0:
  - Adds version with support for external input (i.e., IPSID).
- v1.1.0:
  - Automatically does the necessary mean-removal preprocessing for input neural/behavior data. Automatically adds back the learned means to predicted signals. 
- v1.0.6:
  - Adds option to return the state prediction/filtering error covariances from the Kalman filter.
- v1.0.5:
  - Fixes the n1=0 case for trial based usage.
  - Adds graceful handling of data segments that are too short.
- v1.0.4:
  - Updates readme
- v1.0.3:
  - Fixes readme links on https://pypi.org/project/PSID
- v1.0.2:
  - Changes path to example script. Adds jupyter notebook version.
- v1.0.1:
  - Updates [source/PSID/example/PSID_example.py](https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/PSID_example.py) with smaller file by generating data in code.
- v1.0.0:
  - Adds PSID to pip.