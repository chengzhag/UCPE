# Benchmarks

TODO: old benchmarks! Need to create new ones.

Environment:
- CPU: Intel i9-9960X CPU @ 3.10 GHz
- GPU: Nvidia Qudaro RTX 8000
- RAM: 128 GB

RAM usage doesn't exceed 20 GB, GPU memory usage at batch 64 is roughly 45 GB.

**Transform**|** Type**|** Method**|** Input Size**|** Output Size**|** Batch Size**|** Time (seconds)**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
cube2equi| numpy| default| 256x256| 480x240| 1| 0.0360
cube2equi| numpy| default| 256x256| 480x240| 4| 0.1180
cube2equi| numpy| default| 256x256| 480x240| 16| 0.4512
cube2equi| numpy| default| 256x256| 480x240| 32| 1.0650
cube2equi| numpy| default| 256x256| 480x240| 64| 2.1363
cube2equi| torch| default| 256x256| 480x240| 1| 0.0218
cube2equi| torch| default| 256x256| 480x240| 4| 0.0199
cube2equi| torch| default| 256x256| 480x240| 16| 0.0228
cube2equi| torch| default| 256x256| 480x240| 32| 0.0335
cube2equi| torch| default| 256x256| 480x240| 64| 0.0438
equi2cube| numpy| default| 4000x2000| 256x256| 1| 0.1174
equi2cube| numpy| default| 4000x2000| 256x256| 4| 0.4759
equi2cube| numpy| default| 4000x2000| 256x256| 16| 1.8907
equi2cube| numpy| default| 4000x2000| 256x256| 32| 4.9468
equi2cube| numpy| default| 4000x2000| 256x256| 64| 10.0229
equi2cube| torch| default| 4000x2000| 256x256| 1| 0.0155
equi2cube| torch| default| 4000x2000| 256x256| 4| 0.0328
equi2cube| torch| default| 4000x2000| 256x256| 16| 0.0940
equi2cube| torch| default| 4000x2000| 256x256| 32| 0.1698
equi2cube| torch| default| 4000x2000| 256x256| 64| 0.3200
equi2equi| numpy| default| 4000x2000| 640x320| 1| 2.4446
equi2equi| numpy| default| 4000x2000| 640x320| 4| 9.8693
equi2equi| numpy| default| 4000x2000| 640x320| 16| 42.6679
equi2equi| numpy| default| 4000x2000| 640x320| 32| 96.5504
equi2equi| numpy| default| 4000x2000| 640x320| 64| 193.8804
equi2equi| torch| default| 4000x2000| 640x320| 1| 0.1816
equi2equi| torch| default| 4000x2000| 640x320| 4| 0.5867
equi2equi| torch| default| 4000x2000| 640x320| 16| 2.5047
equi2equi| torch| default| 4000x2000| 640x320| 32| 4.4535
equi2equi| torch| default| 4000x2000| 640x320| 64| 8.7202
equi2pers| numpy| default| 4000x2000| 640x480| 1| 0.0734
equi2pers| numpy| default| 4000x2000| 640x480| 4| 0.2994
equi2pers| numpy| default| 4000x2000| 640x480| 16| 1.1730
equi2pers| numpy| default| 4000x2000| 640x480| 32| 2.7934
equi2pers| numpy| default| 4000x2000| 640x480| 64| 5.4712
equi2pers| torch| default| 4000x2000| 640x480| 1| 0.0026
equi2pers| torch| default| 4000x2000| 640x480| 4| 0.0084
equi2pers| torch| default| 4000x2000| 640x480| 16| 0.0293
equi2pers| torch| default| 4000x2000| 640x480| 32| 0.0447
equi2pers| torch| default| 4000x2000| 640x480| 64| 0.0770


---

CSV:
```
Transform, Type, Method, Input Size, Output Size, Batch Size, Time (seconds)
cube2equi, numpy, default, 256x256, 480x240, 1, 0.0360
cube2equi, numpy, default, 256x256, 480x240, 4, 0.1180
cube2equi, numpy, default, 256x256, 480x240, 16, 0.4512
cube2equi, numpy, default, 256x256, 480x240, 32, 1.0650
cube2equi, numpy, default, 256x256, 480x240, 64, 2.1363
cube2equi, torch, default, 256x256, 480x240, 1, 0.0218
cube2equi, torch, default, 256x256, 480x240, 4, 0.0199
cube2equi, torch, default, 256x256, 480x240, 16, 0.0228
cube2equi, torch, default, 256x256, 480x240, 32, 0.0335
cube2equi, torch, default, 256x256, 480x240, 64, 0.0438
equi2cube, numpy, default, 4000x2000, 256x256, 1, 0.1174
equi2cube, numpy, default, 4000x2000, 256x256, 4, 0.4759
equi2cube, numpy, default, 4000x2000, 256x256, 16, 1.8907
equi2cube, numpy, default, 4000x2000, 256x256, 32, 4.9468
equi2cube, numpy, default, 4000x2000, 256x256, 64, 10.0229
equi2cube, torch, default, 4000x2000, 256x256, 1, 0.0155
equi2cube, torch, default, 4000x2000, 256x256, 4, 0.0328
equi2cube, torch, default, 4000x2000, 256x256, 16, 0.0940
equi2cube, torch, default, 4000x2000, 256x256, 32, 0.1698
equi2cube, torch, default, 4000x2000, 256x256, 64, 0.3200
equi2equi, numpy, default, 4000x2000, 640x320, 1, 2.4446
equi2equi, numpy, default, 4000x2000, 640x320, 4, 9.8693
equi2equi, numpy, default, 4000x2000, 640x320, 16, 42.6679
equi2equi, numpy, default, 4000x2000, 640x320, 32, 96.5504
equi2equi, numpy, default, 4000x2000, 640x320, 64, 193.8804
equi2equi, torch, default, 4000x2000, 640x320, 1, 0.1816
equi2equi, torch, default, 4000x2000, 640x320, 4, 0.5867
equi2equi, torch, default, 4000x2000, 640x320, 16, 2.5047
equi2equi, torch, default, 4000x2000, 640x320, 32, 4.4535
equi2equi, torch, default, 4000x2000, 640x320, 64, 8.7202
equi2pers, numpy, default, 4000x2000, 640x480, 1, 0.0734
equi2pers, numpy, default, 4000x2000, 640x480, 4, 0.2994
equi2pers, numpy, default, 4000x2000, 640x480, 16, 1.1730
equi2pers, numpy, default, 4000x2000, 640x480, 32, 2.7934
equi2pers, numpy, default, 4000x2000, 640x480, 64, 5.4712
equi2pers, torch, default, 4000x2000, 640x480, 1, 0.0026
equi2pers, torch, default, 4000x2000, 640x480, 4, 0.0084
equi2pers, torch, default, 4000x2000, 640x480, 16, 0.0293
equi2pers, torch, default, 4000x2000, 640x480, 32, 0.0447
equi2pers, torch, default, 4000x2000, 640x480, 64, 0.0770
```
