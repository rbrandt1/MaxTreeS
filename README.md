# Efficient binocular stereo correspondence matching with 1-D Max-Trees

## Getting Started

To run the implementation, perform the subsequently listed steps.

### Edit parameters

The parameters of the method are assumed to be kept constant. Therefore, values are assigned to them in code. 
The parameters of the method are assigned in the file `alg-stereoMaxTree/index.cpp` in the region marked 'Parameters'.

### Execute the implementation

#### Middlebury dataset

The implementation expects the following folder structure and files to run on the Middlebury dataset:

```
alg-stereoMaxTree
trainingF/<Testcase>/im0.png (Left image of stereo pair)
trainingF/<Testcase>/im1.png (Right image of stereo pair)
trainingF/<Testcase>/calib.txt (Text file containing a line indicating the maximum disparity in the image pair, i.e. ndisp=<int>)

```

When the said files and folders are in place, the implementation can be run by running the command

```
sudo ./run_middleburry <Testcase>
```

A disparity map with filename ``disp0MaxTreeS_s.pfm`` will be written in the ``trainingF/<Testcase>/`` folder. 
The runtime will be printed in the console.


#### Trimbot2020 dataset

The implementation expects the following folder structure and files to run on the trimbot dataset:

```
alg-stereoMaxTree
trimbotF/<Dataset folder>/vcam_0_f<image number>_undist.png (Left image of stereo pair)
trimbotF/<dataset folder>/vcam_1_f<image number>_undist.png (Right image of stereo pair)

```

When the said files and folders are in place, the implementation can be run on all images in <Dataset folder> by running the command

```
sudo ./run_trimbot <Dataset folder>
```

Disparity maps with filenames ``<image number>.pfm`` will be written in the ``trimbotF/<Dataset>/`` folder. 
The runtime will be printed in the console.


## Dependencies

The implementation is dependent on the following.

* ``opencv/opencv`` (tested on version 4.1.0)
* ``Iorethan/opencv_pfm``
* ``FreeImage``
