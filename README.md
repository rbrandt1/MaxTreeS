# Efficient binocular stereo correspondence matching with 1-D Max-Trees

## Getting Started

To run the implementation, perform the subsequently listed steps.

### Edit parameters

The parameters of the method are assumed to be kept constant. Therefore, values are assigned to them in code. 
The parameters of the method are assigned in the file `matching.cpp`.

### Execute the implementation

The implementation expects the following folder structure and files:

```
alg-stereoMaxTree
trainingF/<Testcase>/im0.png (Left image of stereo pair)
trainingF/<Testcase>/im1.png (Right image of stereo pair)
trainingF/<Testcase>/calib.txt (Text file containing a line indicating the maximum disparity in the image pair, i.e. ndisp=<int>)

```

When the said files and folders are in place, the implementation can be ran by running the command

```
sudo ./run <Testcase>
```

A disparity map with filename ``disp0MaxTreeS_s.pfm`` will be written in the ``trainingF/<Testcase>/`` folder. The runtime will be printed in the console.

## Dependencies

⋅⋅* ``opencv/opencv`` (tested on version 4.1.0)
⋅⋅* ``Iorethan/opencv_pfm``
⋅⋅* FreeImage
