Python Implementation of Graph SLAM
===================================

PyGraphSLAM is my basic implementation of graph SLAM in Python.

### Cherry Picked GIF
![Cherry Picked GIF](https://raw.githubusercontent.com/goktug97/PyGraphSLAM/master/slam.gif)

### Notes
- Trying to improve accuracy, currently the code looks like a scratch book.
- Currently, the loop closure is really bad and not working reliably.

### Requirements
* g2opy             https://github.com/uoip/g2opy

### Usage

``` bash
python slam.py --dataset intel --seed 123123
```

### License
PyGraphSLAM is licensed under the MIT License.

-The `icp.py` is licensed under Apache License, Version 2.0

Original version can be found here: https://github.com/ClayFlannigan/icp

Modified icp to reject some pairs.

### Datasets
http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php
