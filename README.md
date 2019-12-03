# Efficient binocular stereo correspondence matching with 1-D Max-Trees

## Getting Started

To run the implementation, perform the subsequently listed steps.

### Download data sets

Download one or more of the folowing data sets:
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
The parameters of the method are assigned in the file `benchmark.cpp` in run() function calls'.

### Execute implementation

The implementation can be run by executing the command

```
sudo ./main [all,middleburry,kitti2015,synthgarden] [both,metric,result]
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
