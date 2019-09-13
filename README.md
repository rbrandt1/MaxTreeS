# Efficient and Accurate Depth Estimation with 1-D Max-Tree Matching

## Getting Started

To run the implementation, perform the subsequently listed steps.

### Edit parameters

The parameters of the method are assumed to be kept constant. Therefore, values are assigned to them in code. 
The parameters of the method are assigned in the file `benchmark.cpp` in run() function calls'.

### Execute implementation

The implementation can be run by executing the command

```
sudo ./main [all,middleburry,kitti2015,realgarden,synthgarden,driving,monkaa,flyingthings] [both,metric,result]
```

The first argument of the run command specifies on which data set the method should be run. When ``all`` is specified, the method is run on all data sets.
When ``result`` is added to the run command, disparity maps are generated. When ``metric`` is added to the run command, disparity maps are evaluated. 
When ``both`` is added to the run command, disparity maps are both generated and evaluated. 
The accuracy of produced disparity maps, as well as runtime information,  will be printed on the screen and stored in CSV files.

## Dependencies

The implementation is dependent on the following.

* ``opencv/opencv`` (tested on version 4.1.0)
* ``Iorethan/opencv_pfm``
* ``FreeImage``
