Python Implementation of Graph SLAM
===================================

PyGraphSLAM is my basic implementation of graph SLAM in Python.

### GIF
![GIF](https://raw.githubusercontent.com/goktug97/PyGraphSLAM/master/pygraphslam.gif)

### Usage
You can download datasets from http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php

``` bash
python -m src.slam --input intel.clf
```

- If you have nix:

```bash
nix run . -- --input intel.clf
```

### License
PyGraphSLAM is licensed under the MIT License.

-The `icp.py` is licensed under Apache License, Version 2.0

Original version can be found here: https://github.com/ClayFlannigan/icp

Modified icp to reject some pairs.

### Datasets
http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php
