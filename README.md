# MTStereo

Three versions of MTStereo are stored in this repository:

* Version 1.0, implementing a method proposed in the paper entitled "Efficient binocular stereo correspondence matching with 1-D Max-Trees" is stored in the folder ``MaxTreeS-Version_1.0``.
* Version 2.0, implementing a method proposed in the paper entitled "Efficient and Accurate Depth Estimation with 1-D Max-Tree Matching" is stored in the folder ``MaxTreeS-version_2.0``.
* A ROS-node version of MTStereo optimized for scenes containing plants is stored in the folder entitled ``MaxTreeS-version_ROS``.

---
# Version 1.0: Efficient binocular stereo correspondence matching with 1-D Max-Trees

## Getting Started

To run the implementation, perform the subsequently listed steps.

### Download data sets

Download one or more of the following data sets:
* Middlebury - http://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip
* Kitti2015 - http://www.cvlibs.net/download.php?file=data_scene_flow.zip
* Synthgarden - https://gitlab.inf.ed.ac.uk/3DRMS/Challenge2018/tree/master/training

The implementation expects the following folder structure and files to run...

... on the Middlebury dataset:

```
datasets/middlebry/trainingF/<Testcase>/im0.png (Left image of stereo pair)
datasets/middlebry/trainingF/<Testcase>/im1.png (Right image of stereo pair)
datasets/middlebry/trainingF/<Testcase>/disp0GT.pfm (Ground truth)

```

... on the Kitti2015 dataset:

```
datasets/kitti2015/training/image_2/<ImageID>.png (Left image of stereo pair)
datasets/kitti2015/training/image_3/<ImageID>.png (Right image of stereo pair)
datasets/kitti2015/training/disp_noc/<ImageID>.png (Ground truth)

```
... on the Synthgarden dataset:

```
datasets/synthgarden/training/<subset>/vcam_0/vcam_0_<ImageID>_undist.png (Left image of stereo pair)
datasets/synthgarden/training/<subset>/vcam_1/vcam_1_<ImageID>_undist.png (Right image of stereo pair)
datasets/synthgarden/training/<subset>/vcam_0/vcam_0_<ImageID>_dmap.bin (Ground truth)

```


### Edit parameters

The parameters of the method are assumed to be kept constant. Therefore, values are assigned to them in code. 
The parameters of the method are assigned in the files `core.cpp` (in the section entitled Parameters), and `benchmark.cpp` (in `run()` function calls).

### Execute implementation

The implementation can be run by executing the command

```
sudo ./main [all,middlebury,kitti2015,synthgarden] [both,metric,result]
```

The first argument of the run command specifies on which data set the method should be run. When ``all`` is specified, the method is run on all data sets.
When ``result`` is added to the run command, disparity maps are generated. When ``metric`` is added to the run command, disparity maps are evaluated.
When ``both`` is added to the run command, disparity maps are both generated and evaluated. 
The accuracy of produced disparity maps, as well as runtime information,  will be printed on the screen and stored in CSV files.
Metrics computed on Middlebury are calculated without filtering occluded pixels! 

## Dependencies

The implementation is dependent on the following.

* ``opencv/opencv`` (tested on version 4.1.0)
* ``Iorethan/opencv_pfm``

## Terms of use

When you use our implementation in your research, you should cite the following paper:

```
@article{Brandt2020Efficient,
title = "Efficient binocular stereo correspondence matching with 1-D Max-Trees",
journal = "Pattern Recognition Letters",
year = "2020",
issn = "0167-8655",
doi = "https://doi.org/10.1016/j.patrec.2020.02.019",
url = "http://www.sciencedirect.com/science/article/pii/S0167865520300581",
author = "Rafaël Brandt and Nicola Strisciuglio and Nicolai Petkov and Michael H.F. Wilkinson",
}
```

---
# Version 2.0: Efficient and Accurate Depth Estimation with 1-D Max-Tree Matching

## Getting Started

To run the implementation, perform the subsequently listed steps.

### Download data sets

Download one or more of the following data sets:
* Middlebury - http://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip
* Kitti2015 - http://www.cvlibs.net/download.php?file=data_scene_flow.zip
* Driving - https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
* Monkaa - https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
* Flyingthings3D - https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

The implementation expects the following folder structure and files to run...

... on the Middlebury dataset:

```
datasets/middlebry/trainingF/<Testcase>/im0.png (Left image of stereo pair)
datasets/middlebry/trainingF/<Testcase>/im1.png (Right image of stereo pair)
datasets/middlebry/trainingF/<Testcase>/disp0GT.pfm (Ground truth)

```

... on the Kitti2015 dataset:

```
datasets/kitti2015/training/image_0/<ImageID>.png (Left image of stereo pair)
datasets/kitti2015/training/image_1/<ImageID>.png (Right image of stereo pair)
datasets/kitti2015/training/disp_noc/<ImageID>.png (Ground truth)

```


... on the Driving dataset:

```
datasets/driving/frames_cleanpass/35mm_focallength/<forwards_or_backwards>/fast/left/<ImageID>.png (Left image of stereo pair)
datasets/driving/frames_cleanpass/35mm_focallength/<forwards_or_backwards>/fast/right/<ImageID>.png (Right image of stereo pair)
datasets/driving/disparity/35mm_focallength/<forwards_or_backwards>/fast/left/<ImageID>.pfm (Ground truth)

```

... on the Monkaa dataset:

```
datasets/monkaa/frames_cleanpass/<subset>/left/<ImageID>.png (Left image of stereo pair)
datasets/monkaa/frames_cleanpass/<subset>/right/<ImageID>.png (Right image of stereo pair)
datasets/monkaa/disparity/<subset>/left/<ImageID>.pfm (Ground truth)

```

... on the Flyingthings3D dataset:

```
datasets/flyingthings3D/TEST/<...>/<...>/left/<ImageID>.png (Left image of stereo pair)
datasets/flyingthings3D/TEST/<...>/<...>/right/<ImageID>.png (Right image of stereo pair)
datasets/flyingthings3D/disparity/TEST/<...>/<...>/left/<ImageID>.pfm (Ground truth)

```

### Edit parameters

The parameters of the method are assumed to be kept constant. Therefore, values are assigned to them in code. 
The parameters of the method are assigned in the file `benchmark.cpp` in `run()` function calls'.

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
---
# Version ROS

A ROS-node version of MTStereo optimized for scenes containing plants is stored in the folder ``MaxTreeS-version_ROS``.

## Getting Started

To run the implementation, perform the subsequently listed steps.

### Edit parameters

The parameters of the method are assumed to be kept constant. Therefore, values are assigned to them in code. 
The parameters of the method are assigned in the file `pcl.cpp` in the `setParameters()` and `main()` function.


### Execute implementation

The implementation can be run by executing the command

```
sudo ./main
```

## Dependencies

The implementation is dependent on the following.

* ``opencv/opencv`` (tested on version 4.1.0)
* ``Iorethan/opencv_pfm``
* ``FreeImage``
* ``ROS``

## Example

Examples of point clouds the ROS version of MTStereo can produce can be viewed by watching the following video.

[![Watch the video](https://img.youtube.com/vi/VfYLtVT-DKY/maxresdefault.jpg)](https://youtu.be/VfYLtVT-DKY)