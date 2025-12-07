## equi2pers

equirectangular to perspective transformation


### TODO:

- [x] Crop is slightly different `numpy` and `torch` (FIXED)
- [x] Equi2Pers outputs for `numpy` and `torch` differs a little bit. Need to figure out why this happens. The outputs are the same regardless of the sampling method, so it must be the preprocessing (where the rotation matrix is set). (FIXED)
