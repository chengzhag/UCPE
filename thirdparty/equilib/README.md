<h1 align="center">
  equilib
</h1>

<h4 align="center">
  Processing Equirectangular Images with Python
</h4>

<div align="center">
  <a href="https://badge.fury.io/py/pyequilib"><img src="https://badge.fury.io/py/pyequilib.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pyequilib"><img src="https://img.shields.io/pypi/pyversions/pyequilib"></a>
  <a href="https://github.com/haruishi43/equilib/actions"><img src="https://github.com/haruishi43/equilib/workflows/ci/badge.svg"></a>
  <a href="https://github.com/haruishi43/equilib/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/haruishi43/equilib"></a>
</div>

<img src=".img/equilib.png" alt="equilib" width="720"/>

- A library for processing equirectangular image that runs on Python.
- Developed using Python>=3.6 (`c++` is WIP).
- Compatible with `cuda` tensors for faster processing.
- No need for other dependencies except for `numpy` and `torch`.
- Added functionality like creating rotation matrices, batched processing, and automatic type detection.
- Works with various input modals
- Highly modular

If you found this module helpful to your project, please site this repository:
```
@software{pyequilib2021github,
  author = {Haruya Ishikawa},
  title = {PyEquilib: Processing Equirectangular Images with Python},
  url = {http://github.com/haruishi43/equilib},
  version = {0.5.0},
  year = {2021},
}
```

## Installation:

Prerequisites:
- Python (>=3.6)
- Pytorch (tested on 1.12)

```Bash
pip install pyequilib
```

For developing, use:

```Bash
git clone --recursive https://github.com/haruishi43/equilib.git
cd equilib

pip install -r requirements.txt

pip install -e .
# or
python setup.py develop
```

__NOTE__: might not work for PyTorch>=2.0. If you have any issues, please open an issue.

## Basic Usage:

`equilib` has different transforms of equirectangular (or cubemap) images (note each transform has `class` and `func` APIs):
- `Cube2Equi`/`cube2equi`: cubemap to equirectangular transform
- `Equi2Cube`/`equi2cube`: equirectangular to cubemap transform
- `Equi2Equi`/`equi2equi`: equirectangular transform
- `Equi2Pers`/`equi2pers`: equirectangular to perspective transform
- `Pers2Equi`/`pers2equi`: perspective to equirectangular transform

There are no _real_ differences in `class` or `func` APIs:
- `class` APIs will allow instantiating a class which you can call many times without having to specify configurations (`class` APIs call the `func` API)
- `func` APIs are useful when there are no repetitive calls
- both `class` and `func` APIs are extensible, so you can extend them to your use-cases or create a method that's more optimized (pull requests are welcome btw)

Each API automatically detects the input data type (`numpy.ndarray` or `torch.Tensor`), and outputs are the same type. The data layout for input images should be channel-first with dimensions either `BxCxHxW` or `CxHxW`.

The arguments for each `class` or `func` depends on the transform, but here are the common arguments:
- `rots`: used to specify the rotation using three angles [pitch, yaw, roll](https://simple.wikipedia.org/wiki/Pitch,_yaw,_and_roll) in radians.
- `z_down (bool)`: whether to use a coordinate system with z-axis pointing down, defaults to `False`
- `mode (str)`: interpolation mode, defaults to `bilinear`
- `clip_output (bool)`: whether to clip values based on the range of the input values, default to `True`

An example for `Equi2Pers`/`equi2pers`:

<table>
<tr>
<td><pre>Equi2Pers</pre></td>
<td><pre>equi2pers</pre></td>
</tr>

<tr>
<td>
<pre>

```Python
import numpy as np
from PIL import Image
from equilib import Equi2Pers

# Input equirectangular image
equi_img = Image.open("./some_image.jpg")
equi_img = np.asarray(equi_img)
equi_img = np.transpose(equi_img, (2, 0, 1))

# rotations
rots = {
    'roll': 0.,
    'pitch': np.pi/4,  # rotate vertical
    'yaw': np.pi/4,  # rotate horizontal
}

# Intialize equi2pers
equi2pers = Equi2Pers(
    height=480,
    width=640,
    fov_x=90.0,
    mode="bilinear",
)

# obtain perspective image
pers_img = equi2pers(
    equi=equi_img,
    rots=rots,
)
```

</pre>
</td>

<td>
<pre>

```Python
import numpy as np
from PIL import Image
from equilib import equi2pers

# Input equirectangular image
equi_img = Image.open("./some_image.jpg")
equi_img = np.asarray(equi_img)
equi_img = np.transpose(equi_img, (2, 0, 1))

# rotations
rots = {
    'roll': 0.,
    'pitch': np.pi/4,  # rotate vertical
    'yaw': np.pi/4,  # rotate horizontal
}

# Run equi2pers
pers_img = equi2pers(
    equi=equi_img,
    rots=rots,
    height=480,
    width=640,
    fov_x=90.0,
    mode="bilinear",
)
```

</pre>
</td>
</table>

For more information about how each APIs work, take a look in [.readme](.readme/) or go through example codes in the `tests` or `scripts`.


### Coordinate System:

__Right-handed rule XYZ global coordinate system__. `x-axis` faces forward and `z-axis` faces up.
- `roll`: counter-clockwise rotation about the `x-axis`
- `pitch`: counter-clockwise rotation about the `y-axis`
- `yaw`: counter-clockwise rotation about the `z-axis`

You can chnage the right-handed coordinate system so that the `z-axis` faces down by adding `z_down=True` as a parameter.

See demo scripts under `scripts`.


## Grid Sampling

To process equirectangular images fast, whether to crop perspective images from the equirectangular image, the library takes advantage of grid sampling techniques.
Some sampling techniques are already implemented, such as `scipy.ndimage.map_coordiantes` and `cv2.remap`.
This project's goal was to reduce these dependencies and use `cuda` and batch processing with `torch` and `c++` for a faster processing of equirectangular images.
There were not many projects online for these purposes.
In this library, we implement varieties of methods using `c++`, `numpy`, and `torch`.
This part of the code needs `cuda` acceleration because grid sampling is parallelizable.
For `torch`, the built-in `torch.nn.functional.grid_sample` function is very fast and reliable.
I have implemented a _pure_ `torch` implementation of `grid_sample` which is very customizable (might not be fast as the native function).
For `numpy`, I have implemented grid sampling methods that are faster than `scipy` and more robust than `cv2.remap`.
Just like with this implementation of `torch`, `numpy` implementation is just as customizable.
It is also possible to pass the `scipy` and `cv2`'s grid sampling function through the use of `override_func` argument in `grid_sample`.
Developing _faster_ approaches and `c++` methods are __WIP__.
See [here](equilib/grid_sample/README.md) for more info on implementations.

Some notes:

- By default, `numpy`'s [`grid_sample`](equilib/grid_sample/numpy/) will use pure `numpy` implementation. It is possible to override this implementation with `scipy` and `cv2`'s implementation using [`override_func`](tests/equi2pers/numpy_run_baselines.py).
- By default, `torch`'s [`grid_sample`](equilib/grid_sample/torch/) will use the official implementation.
- Benchmarking codes are stored in `tests/`. For example, benchmarking codes for `numpy`'s `equi2pers` is located in [`tests/equi2pers/numpy_run_baselines.py`](tests/equi2pers/numpy_run_baselines.py) and you can benchmark the runtime performance using different parameters against `scipy` and `cv2`.

## Develop:

Test files for `equilib` are included under `tests`.

Running tests:
```Bash
pytest tests
```

Note that I have added codes to benchmark every step of the process so that it is possible to optimize the code.
If you find there are optimal ways of the implementation or bugs, all pull requests and issues are welcome.

Check [CONTRIBUTING.md](./CONTRIBUTING.md) for more information

### TODO:

- [ ] Documentations for each transform
- [x] Add table and statistics for speed improvements
- [x] Batch processing for `numpy`
- [x] Mixed precision for `torch`
- [ ] `c++` version of grid sampling
- [ ] More accurate intrinsic matrix formulation using vertial FOV for `equi2pers`
- [ ] Multiprocessing support (slow when running on `torch.distributed`)

## Acknowledgements:

- [py360convert](https://github.com/sunset1995/py360convert)
- [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular)
