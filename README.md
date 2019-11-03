Python Implementation of Graph SLAM
===================================

PyGraphSLAM is my basic implementation of graph SLAM in Python.

### GIF
![GIF](https://raw.githubusercontent.com/goktug97/PyGraphSLAM/master/graphslam.gif)

### Notes
- Trying to improve accuracy, currently the code looks like a scratch book.
- Currently, the loop closure is really bad and not working reliably.

### Requirements
* g2opy             https://github.com/uoip/g2opy

### Usage
You can download datasets from http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php

``` bash
python slam.py --input intel.clf
```

For more options
```bash
python slam.py --help
```

### License
PyGraphSLAM is licensed under the MIT License.

-The `icp.py` is licensed under Apache License, Version 2.0

Original version can be found here: https://github.com/ClayFlannigan/icp

Modified icp to reject some pairs.

### Datasets
http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php
