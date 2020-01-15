// GENERAL
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <string>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <numeric>
#include <queue>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// ROS
#include "ros/ros.h"
#include "ros/package.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/fill_image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "stereo_msgs/DisparityImage.h"


/// PCL
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include "tf2_msgs/TFMessage.h"




/****** Definitions & Parameters ******/

#define ROOT				(-1)

ros::Publisher pub_pcl;

typedef struct MaxNode {
 int parent;
  int Area;
  bool levelroot;
  int dispL;
  int dispR;
  int dispLprev;
  int dispRprev;
  uchar nChildren;
  int begIndex;
  int matchId;
  bool curLevel;
  bool prevCurLevel;
}
MaxNode;

typedef struct Match {
  float cost;
  int match;
}
Match;

typedef cv::Vec < float, 2 > Vec2f;

int image_width {0}, image_height {0};
int crop_N {0}, crop_E {0}, crop_S {0}, crop_W {0};
int subsample_pointcloud;
int minAreaMatchedFineTopNodeL;
int maxAreaMatchedFineTopNodeL;
int nColors;
int minDisp;
int maxDisp;
float maxDist;
float minDist;
ros::Publisher chatter_pub;
ros::Publisher chatter_pub2;
bool performCropping;
int minAreaMatchedFineTopNode;
int maxAreaMatchedFineTopNode;
uint kernelCostComp;
uint kernelCostCompFinalStep;
int kernelCostCompFinalFilter;
bool sparse;
int everyNthRow;
int nThreads;

cv::Mat costVolume[300];

int thInterestingOne;
float uniquenessRatio;
int costVolumeFilterSize;
int noiseFilterKernel;

float alpha;

/****** Functions ******/


float contextCost(int indexL, int indexR, MaxNode* treeHorL, MaxNode* treeHorR) {

	int maxiter = 2;

	if(indexL >= 0 && indexR >= 0){

		float total = 1;

		float sum = 0;

		float weight = 1;

		while (indexL != ROOT && indexR != ROOT && total < maxiter) {
			sum += weight * abs(treeHorL[indexL].Area - treeHorR[indexR].Area);
			total += weight;
			indexL = treeHorL[indexL].parent;
			indexR = treeHorR[indexR].parent;
		}
		return (sum / total);
	}else{
		return std::numeric_limits<int>::max();
	}
}

cv::Mat CalcRGBmax(cv::Mat i_RGB){
    std::vector<cv::Mat> planes(3);
    cv::split(i_RGB, planes);
    return cv::Mat(cv::max(planes[2], cv::max(planes[1], planes[0])));
}



class BuildCostVolumeParallel : public cv::ParallelLoopBody{
public:
  BuildCostVolumeParallel (const int kernelSize, const cv::Mat imlL, const cv::Mat imr, const int maxDisp )  : kernelSize(kernelSize),imlL(imlL),imr(imr), maxDisp(maxDisp){}
  virtual void operator()(const cv::Range & range) const {
    for (int d = range.start; d < range.end; d++){
	costVolume[d] =	abs(imlL-imr(cv::Rect(maxDisp-d,0,imr.cols-maxDisp,imr.rows)));
	cv::cvtColor(costVolume[d], costVolume[d], cv::COLOR_RGB2GRAY);
	costVolume[d].convertTo(costVolume[d],CV_32F);

	cv::GaussianBlur(costVolume[d],costVolume[d],cv::Size(kernelSize,kernelSize),0);
     }
  }
  BuildCostVolumeParallel& operator=(const BuildCostVolumeParallel &) {
    return *this;
  };
private:
  const int kernelSize;
  const int maxDisp;
  const cv::Mat imlL;
  const cv::Mat imr;
};





class NthChildrenSide : public cv::ParallelLoopBody{
public:
  NthChildrenSide ( std::vector<std::vector<int> >& topsL, int n, std::vector<std::vector<int> >* topsLnTh, MaxNode* treeLeftHor )  : topsL(topsL),n(n), topsLnTh(topsLnTh), treeLeftHor(treeLeftHor){}
  virtual void operator()(const cv::Range & range) const {
    for (int r = range.start; r < range.end; r++){
	
		// get unique nth tops

		std::vector<int> LevelNChildren;
		for (uint c = 0; c < topsL[r].size(); c++) {
			int current = topsL[r][c];

			for (int iter = 0; current != ROOT && iter < n;
					iter++) {
				current = treeLeftHor[current].parent;

				while (current != ROOT && !treeLeftHor[current].levelroot) {
					current = treeLeftHor[current].parent;
				}
			}

			if (current != ROOT)
				LevelNChildren.push_back(current);
		}

		if (LevelNChildren.size() > 0) {
			sort(LevelNChildren.begin(), LevelNChildren.end());
			LevelNChildren.erase(unique(LevelNChildren.begin(), LevelNChildren.end()), LevelNChildren.end());
		}

		// push all children

		std::vector<int> allChildren;
		for (uint c = 0; c < LevelNChildren.size(); c++) {
			int current = LevelNChildren[c];
			current = treeLeftHor[current].parent;

			while (current != ROOT) {
				if (treeLeftHor[current].levelroot)
					allChildren.push_back(current);

				current = treeLeftHor[current].parent;
			}
		}

		std::vector<int> rowLNew;

		for (uint c = 0; c < LevelNChildren.size(); c++) {
			bool found = false;
			for (uint i = 0; i < allChildren.size(); i++) {
				if(LevelNChildren[c] == allChildren[i]){
					found = true;
					break;
				}
			}
			if(!found){
				rowLNew.push_back(LevelNChildren[c]);
				treeLeftHor[LevelNChildren[c]].curLevel=true;
			}
		}
		LevelNChildren.clear();
		allChildren.clear();

		topsLnTh->at(r) = rowLNew;
     }
  }
  NthChildrenSide& operator=(const NthChildrenSide &) {
    return *this;
  };
private:
 std::vector<std::vector<int> >& topsL;
 int n;
 std::vector<std::vector<int> >* topsLnTh;
 MaxNode* treeLeftHor ;
};





/**
 *
 * Compute the nth-level coarse top nodes in Max-Trees
 *
 * @param  std::vector<std::vector<int>> topsR The fine top nodes in the right image.
 * @param  std::vector<std::vector<int>> topsL The fine top nodes in the left image.
 * @param  int n The level
 * @param  std::vector<std::vector<int>> * topsRnTh Returns the nth-level coarse top nodes in the right Max-Tree
 * @param  std::vector<std::vector<int>> * topsLnTh Returns the nth-level coarse top nodes in the left Max-Tree
 * @param  MaxNode * treeRightHor Right Max-Tree
 * @param  MaxNode * treeLeftHor Left Max-Tree
 * @param  int height Height of the input images.
 *
 */
void getNthTops(std::vector<std::vector<int>> topsR,
		std::vector<std::vector<int>> topsL, int n,
		std::vector<std::vector<int>> * topsRnTh,
		std::vector<std::vector<int>> * topsLnTh, MaxNode * treeRightHor,
		MaxNode * treeLeftHor, int height) {

	NthChildrenSide nthChildrenSide(topsL, n, topsLnTh, treeLeftHor);
	cv::parallel_for_(cv::Range(0, height), nthChildrenSide);

	NthChildrenSide nthChildrenSide2(topsR, n, topsRnTh, treeRightHor);
	cv::parallel_for_(cv::Range(0, height), nthChildrenSide2);

}



void buildCostVolume(cv::Mat iml,cv::Mat imr, int maxDisp,int kernelSize){
	
	cv::Rect roiL = cv::Rect(maxDisp,0,iml.cols-maxDisp,imr.rows);
	cv::Mat imlL = iml(roiL);

	BuildCostVolumeParallel buildCostVolumeParallel(kernelSize, imlL, imr, maxDisp);
	cv::parallel_for_(cv::Range(0, maxDisp), buildCostVolumeParallel);
}



/**
 *
 * Extract largest node containing branch
 *
 * @param    MaxNode * tree  MaxTree.
 * @param    int index Index of fine top node.
 *
 */

int getNextParent(MaxNode * tree, int index){
  index = tree[index].parent;
  while(index != ROOT && tree[index].parent != ROOT && !tree[index].levelroot){
    index = tree[index].parent;
  }
  return index;
}

int getInterestingOne(MaxNode * tree, int index) {
    int nextIndex = getNextParent(tree,index);
    while(index!= ROOT && nextIndex != ROOT && abs(tree[nextIndex].Area - tree[index].Area ) <= thInterestingOne){
        index = getNextParent(tree,index);
        nextIndex = getNextParent(tree,index);
    }
    return index;
}

/**
 *
 * Clear (i.e. deallocate the memory) of a 2D vector
 *
 * @param    std::vector<std::vector<int>> vect  The vector to clear.
 *
 */
void clear2Dvect(std::vector < std::vector < int >> vect) {
  for (uint i = 0; i < vect.size(); i++) {
    vect[i].clear();
  }
  vect.clear();
}


int neighbourHelp(MaxNode * tree, int index,int width,int height,bool useCur){
  if (index < 0 || index >= width * height)
    return ROOT;

  if (useCur) {
    while (index != ROOT && !tree[index].curLevel) {
      index = tree[index].parent;
    }
  } else {
    while (index != ROOT && !tree[index].curLevel) {
      index = tree[index].parent;
    }
  }

  return index;

}



void pub_pp(const sensor_msgs::ImageConstPtr & img_msg_l, cv::Mat img, cv::Mat colors,
  const sensor_msgs::CameraInfoConstPtr & camInfo) {


  auto tmpP = camInfo->P;

  float center_x_ = tmpP[2];
  float center_y_ = tmpP[6];
  float focal_length_ = tmpP[0];
  float baseline_m = 23.388874828/focal_length_;
  

  pcl::PointCloud < pcl::PointXYZRGB > ::Ptr cloud(
    new pcl::PointCloud < pcl::PointXYZRGB > );
  cloud->header.frame_id = "cam_0_optical_frame"; 

  /// The useful region of the disparity map is cropped

  for (float x = 0; x < colors.cols; x += 1) {
    for (float y = 0; y < colors.rows; y += 1) {

      float v = img.at < float > (y, x);

      float z_value = baseline_m * focal_length_ / v;

      if (z_value > maxDist || z_value < minDist  ) {
        continue;
      }

      pcl::PointXYZRGB pt;

      pt.x = (x - center_x_) * (z_value / focal_length_);
      pt.y = (y - center_y_) * (z_value / focal_length_);
      pt.z = z_value; 

      pt.r = colors.at < cv::Vec3f > (y, x)[2];
      pt.g = colors.at < cv::Vec3f > (y, x)[1];
      pt.b = colors.at < cv::Vec3f > (y, x)[0];

      cloud->points.push_back(pt);
    }
  }

  pub_pcl.publish(cloud);

  ROS_INFO("Published point cloud");
}




/**
 *
 * Get the neighbors below (in terms of y coordinate) a node
 *
 * @param   MaxNode * tree Max-tree
 * @param	int indexCenter Index of node
 * @param	int width Width of image
 * @param	int height Height of image
 * @param	bool useCur Use current level or that of previous iteration.
 * @param	bool dispCheck Whether only nodes with a disparity assigned to them are included.
 * @param	int kernelCostComp At most the kernelCostComp nodes above/below the node are included.
 * @return	 The neighbors below (in terms of y coordinate) a node
 *
 */
int getNeighborsBottom(MaxNode * tree, int indexCenter, int width, int height) {
	
	if (indexCenter < 0 || indexCenter >= width * height)
		return -1;

	int index = tree[indexCenter].begIndex + tree[indexCenter].Area / 2;

	index = index - width;

	if (index < 0 || index >= width * height){
		return -1;
	}else{
		if(!tree[index].levelroot)
			index = tree[index].parent;
		return index;
	}
}

/**
 *
 * Get the neighbors above (in terms of y coordinate) a node
 *
 * @param   MaxNode * tree Max-tree
 * @param	int indexCenter Index of node
 * @param	int width Width of image
 * @param	int height Height of image
 * @param	bool useCur Use current level or that of previous iteration.
 * @param	bool dispCheck Whether only nodes with a disparity assigned to them are included.
 * @param	int kernelCostComp At most the kernelCostComp nodes above/below the node are included.
 * @return	 The neighbors below (in terms of y coordinate) a node
 *
 */
int getNeighborsTop(MaxNode * tree, int indexCenter, int width, int height) {
	
	if (indexCenter < 0 || indexCenter >= width * height)
		return -1;

	int index = tree[indexCenter].begIndex + tree[indexCenter].Area / 2;

	index = index + width;

	if (index < 0 || index >= width * height){
		return -1;
	}else{
		if(!tree[index].levelroot)
			index = tree[index].parent;
		return index;
	}
}



/**
 *
 * Compute the fine top nodes in Max-Trees
 *
 * @param    MaxNode* treeLeftHor Left Max-tree
 * @param    MaxNode* treeRightHor Right Max-tree
 * @param    int width Width of input images
 * @param    int height Height of input images
 * @param    std::vector<std::vector<int>> *topsL Will store the indexes of the fine top nodes in the left image.
 * @param    std::vector<std::vector<int>> *topsR Will store the indexes of the fine top nodes in the right image.
 *
 */
void getTops( MaxNode * treeLeftHor, int width,
		int height, std::vector<std::vector<int>> *topsL) {

	for (int r = 0; r < height; r++) {
		std::vector<int> rowL;
	

		for (int c = 0; c < width; c++) {
			int index = c + (height - 1 - r) * width;
	
			if (treeLeftHor[index].levelroot
					&& treeLeftHor[index].nChildren <= 1) {
						if(treeLeftHor[index].Area > 1 && treeLeftHor[index].Area < 200){
							rowL.push_back(index);
							treeLeftHor[index].curLevel=true;
							treeLeftHor[index].prevCurLevel=true;
						}
					
			}
		}

		(*topsL).push_back(rowL);
	
	}
}

class BuildTreeParalel: public cv::ParallelLoopBody {

  private: int height;
  int width;
  cv::Mat gval;
  MaxNode * node;
public:

  BuildTreeParalel(int height, int width, cv::Mat gval, MaxNode * node): height(height),
  width(width),
  gval(gval),
  node(node) {}

  virtual void operator()(const cv::Range & range) const {
    for (int row = range.start; row < range.end; row++) {
      int current, i, next, curr_parent;
      int invRow = height - 1 - row;

      for (i = 0, current = row * width; i < width; i++, current++) {
        node[current].parent = ROOT;
        node[current].Area = 1;
        node[current].levelroot = true;
        node[current].nChildren = 0;
        node[current].begIndex = std::numeric_limits < int > ::max();
	node[current].dispL = -1;
	node[current].dispR = -1;

        node[current].matchId = -1;
        node[current].curLevel = false;
        node[current].curLevel = false;
      }

      current = row * width; // reset the counter

      for (i = 1, next = current + 1; i < width; i++, next++) {
        /*
         * Checking the gray values of the nodes.
         * If the gray value of the current node is equal to the gray value
         * of the next node, the next node is a children of the current node and
         * the current area is augmented by 1.
         * If the gray value of the current node is lower than the gray value
         * of the next node, the next node is a children of the current node.
         * Since a new level of intensity is found, it becomes the new current node.
         * If the gray value of the current node is greater than the gray value
         * of the next node, the current node should be a children of the next node.
         * This should be repeated for every parent of the current node.
         */
        if (gval.at < float > (invRow, current % width) <= gval.at < float > (invRow, next % width)) {
          /* ascending or flat */

          node[next].parent = current;

          if (gval.at < float > (invRow, current % width) == gval.at < float > (invRow, next % width)) {
            /*flat */
            node[current].Area++;
            node[next].levelroot = false;
          } else {
            current = next; /* new top level root */
          }
        } else {
          /* descending */
          curr_parent = node[current].parent; // save the current node parent

          /*
           * For each parent of the current node, until a root node is reached or
           * when the gray value of the current node becomes lower, this procedure
           * of swapping parents should be repeated.
           */
          while ((curr_parent != ROOT) && (gval.at < float > (invRow, curr_parent % width) > gval.at < float > (invRow, next % width))) {
            node[curr_parent].Area += node[current].Area;
            current = curr_parent;
            curr_parent = node[current].parent;
          }
          node[current].parent = next;
          if (gval.at < float > (invRow, current % width) == gval.at < float > (invRow, node[current].parent % width) && next != ROOT)
            node[current].levelroot = false;
          node[next].Area += node[current].Area;
          node[next].parent = curr_parent;
          if (gval.at < float > (invRow, node[next].parent % width) == gval.at < float > (invRow, next % width) && curr_parent != ROOT)
            node[next].levelroot = false;
          current = next;
        }
      }

      /*
       * Go through the root path to update the area value of the root node.
       */
      curr_parent = node[current].parent;
      while (curr_parent != ROOT) {
        node[curr_parent].Area += node[current].Area;
        current = curr_parent;
        curr_parent = node[current].parent;
      }

      for (i = row * width; i < (row + 1) * width; i++) {
        int current = i;
        if (node[current].levelroot && node[current].parent != ROOT) {
          while (!node[node[current].parent].levelroot) {
            current = node[current].parent;
          }
          node[node[current].parent].nChildren++;
        }
      }

      for (i = row * width; i < (row + 1) * width; i++) {
        int current = i;
        if (node[current].nChildren == 0 && node[current].levelroot) {
          int smallest = current;
          while (node[current].parent != ROOT) {
            if (current < smallest) {
              smallest = current;
            }
            if (node[current].begIndex > smallest) {
              node[current].begIndex = smallest;
            }
            current = node[current].parent;
          }
        }
      }
    }
  }
};


cv::Mat edgeFilter(cv::Mat img){

	int morph_size = 1;
	int morph_elem = 0;
  cv::Mat element = getStructuringElement( morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

  cv::morphologyEx(img, img, cv::MORPH_OPEN , element);

  return img;
}



/**
 *
 * Build a max tree from an image
 *
 * @param    cv::Mat gval The image
 * @return   The generated Max-Tree
 *
 */
MaxNode * buildTree(cv::Mat gval) {
  int height = gval.rows;
  int width = gval.cols;
  MaxNode * node = (MaxNode * ) malloc(width * height * sizeof(MaxNode));

  parallel_for_(cv::Range(0, height), BuildTreeParalel(height, width, gval, node));

  return node;
}





class ParallelMatching : public cv::ParallelLoopBody{
public:
  ParallelMatching (cv::Mat & dispMap, const int width, const int maxDisp, const int minDisp,MaxNode *treeLeftHor,MaxNode *treeRightHor)
  : dispMap(dispMap), width(width), maxDisp(maxDisp), minDisp(minDisp),treeLeftHor(treeLeftHor),treeRightHor(treeRightHor){}
  virtual void operator()(const cv::Range & range) const {
    for (int i = range.start; i < range.end; i++){
	int r = i / dispMap.cols;
	int c = i % dispMap.cols;
	if(c > maxDisp){
		int indexMin_Left = 0;
		int indexMin_Right = 0;
		float bestCost = 10000000;
		float secondBestCost = 10000000;
		for(int d = 0; d < maxDisp; d++){

			float *costData = (float*)(costVolume[d].data);
			float colCost = costData[costVolume[d].cols * (r) + (c-maxDisp)];

			float cost = 0;
			float descendentsCost = 0;
			int width = dispMap.cols;
			int height = dispMap.rows;
			int row = (height - 1 - r)*width;
			int indexL = row+c-d;
			int indexR = row+c;

			if(!treeLeftHor[indexL].levelroot)
				indexL = treeLeftHor[indexL].parent;
			if(!treeRightHor[indexR].levelroot)
				indexR = treeRightHor[indexR].parent;

			if(indexL != ROOT && indexR != ROOT && treeLeftHor[indexL].Area > minAreaMatchedFineTopNode && treeRightHor[indexR].Area > minAreaMatchedFineTopNode &&  treeLeftHor[indexL].Area < maxAreaMatchedFineTopNode && treeRightHor[indexR].Area < maxAreaMatchedFineTopNode){
				int NeighborsLTop = getNeighborsTop(treeLeftHor, indexL, width, height);
				int NeighborsRTop = getNeighborsTop(treeRightHor, indexR,  width, height);
				int NeighborsLBottom = getNeighborsBottom(treeLeftHor, indexL,  width, height);
				int NeighborsRBottom = getNeighborsBottom(treeRightHor, indexR, width, height);


				// context cost aggregation

				float tmp = contextCost(indexL,indexR,treeLeftHor,treeRightHor);
				descendentsCost += tmp;
				
				if(NeighborsLTop!= -1 && NeighborsRTop != -1){	
					tmp = contextCost(NeighborsLTop,NeighborsRTop,treeLeftHor,treeRightHor);
					descendentsCost += tmp;
				}

				if(NeighborsLBottom!= -1 && NeighborsRBottom != -1){
					tmp = contextCost(NeighborsLBottom,NeighborsRBottom,treeLeftHor,treeRightHor);
					descendentsCost += tmp;
				}

				cost += (descendentsCost / 3) * (1 - alpha);


			}

			cost += (colCost)  * alpha;

			if(cost <= bestCost){
				secondBestCost = bestCost;
				bestCost = cost;
				indexMin_Left = d;      
			}
		}
		if(uniquenessRatio/100 > (secondBestCost-bestCost)/bestCost){
			indexMin_Left = 0;
		}else{
	
			bestCost = 10000000;
			secondBestCost = 10000000;
			for(int d = 0; d < maxDisp; d++){
				if(d+indexMin_Left <maxDisp && c-maxDisp+d < width ){
					float *costData = (float*)(costVolume[indexMin_Left+d].data);
					float cost = costData[costVolume[d].cols * (r) + (c-maxDisp+d)];


					if(cost <= bestCost){
						secondBestCost = bestCost;
						bestCost = cost;
						indexMin_Right = d;      
					}
				}
			}


		}
		if(indexMin_Right > 1)
			indexMin_Left = 0;
		if(uniquenessRatio/100 > (secondBestCost-bestCost)/bestCost)
			indexMin_Left = 0;

		dispMap.ptr<float>(r)[c] = indexMin_Left;
	}
    }
  }
  ParallelMatching& operator=(const ParallelMatching &) {
    return *this;
  };
private:
  cv::Mat & dispMap;
  int width;
  int maxDisp;
  MaxNode *treeLeftHor;
  MaxNode *treeRightHor;
  int minDisp;
};






class SpeckleFilter : public cv::ParallelLoopBody{
public:
  SpeckleFilter(float * dispMap,float * dispMapFinal,int cols,int rows, const int kermel, const int th, const int devider)
  :  dispMap(dispMap),dispMapFinal(dispMapFinal),cols(cols),rows(rows), kermel(kermel),th(th),devider(devider){}
  virtual void operator ()(const cv::Range& range) const {
	for (int r = range.start; r < range.end; r++) {
	    for (int c = 0; c < cols; c++) {
	
		float curval = dispMap[cols * (r) + (c)];
		if (curval > 1) {
			int total = 0;
			int totalMean = 0;
			int rBeg = r - kermel > 0 ? r - kermel : r;
			int cBeg = c - kermel > 0 ? c - kermel : c;
			int cEnd = (c + kermel <= cols ? c + kermel : c);
			int rEnd = (r + kermel <= rows ? r + kermel : r);
		  	for (int cOff = cBeg; cOff < cEnd; cOff++) {
				for (int rOff = rBeg; rOff < rEnd; rOff++) {
					
					float val = dispMap[cols * (rOff) + (cOff)];
					if (val > 1) {
					    if (abs(val - curval) <= 6) {
						total++;
					    }
					}
			    	}
			}

			if (total < th){
				for(int i = -1;i <= devider;i++){
					for(int j = -1;j <= devider;j++){
						int index = cols*devider * (r*devider+j) + (c*devider+i);
						if(index >= 0 && index < cols*rows*devider*devider)
							dispMapFinal[index] =-1;
					}
				}
				
			}
			    
		}
	   }
	}
}
SpeckleFilter& operator=(const SpeckleFilter &) {
  return *this;
};
private:
	float * dispMap;
	float * dispMapFinal;
	int cols;
	int rows;
	const int kermel;
	const int th;
	const int devider;
};




class NoiseFilter : public cv::ParallelLoopBody{
public:
  NoiseFilter (float * dispMap,float * dispMapFinal,int cols,int rows, const int kermel)
  :  dispMap(dispMap),dispMapFinal(dispMapFinal),cols(cols),rows(rows), kermel(kermel){}
  virtual void operator ()(const cv::Range& range) const {
	for (int r = range.start; r < range.end; r++) {
	    for (int c = 0; c < cols; c++) {
	
		float curval = dispMap[cols * (r) + (c)];
		if (curval > 1) {
			int total = 0;
			int totalMean = 0;
			int rBeg = r - kermel > 0 ? r - kermel : r;
			int cBeg = c - kermel > 0 ? c - kermel : c;
			int cEnd = (c + kermel <= cols ? c + kermel : c);
			int rEnd = (r + kermel <= rows ? r + kermel : r);
		  	for (int cOff = cBeg; cOff < cEnd; cOff++) {
				for (int rOff = rBeg; rOff < rEnd; rOff++) {
					float val = dispMap[cols * (rOff) + (cOff)];
					if (val > 1) {
					    if (abs(val - curval) <= 6) {
						total++;
					    }
					    else {
						totalMean++;
					    }
					}
			    	}
			}

			if (totalMean != 0 && total < totalMean)
			    dispMapFinal[cols * (r) + (c)] = -1;
		}
	   }
	}
}
NoiseFilter& operator=(const NoiseFilter &) {
  return *this;
};
private:
	float * dispMap;
	float * dispMapFinal;
	int cols;
	int rows;
	const int kermel;
};





 
cv::Mat bilateralFilter(cv::Mat dispMap,int kernel = 3){
	for(int r =0;r<dispMap.rows;r++){
		for(int c =0;c<dispMap.cols;c++){
			if(dispMap.at<float>(r,c) != 0)
				dispMap.at<float>(r,c) = dispMap.at<float>(r,c) + 2550;
		}     
	}

	cv::Mat img2;
	cv::bilateralFilter(dispMap,img2,0,kernel,kernel);
	dispMap = img2;

	for(int r =0;r<dispMap.rows;r++){
		for(int c =0;c<dispMap.cols;c++){
			if(dispMap.at<float>(r,c) != 0)
				dispMap.at<float>(r,c) = dispMap.at<float>(r,c) - 2550;
		}     
	}
	return dispMap;
}


cv::Mat textureFilter(cv::Mat img,cv::Mat dispMap,int kernel,float thr){
	cv::Mat tmp;
	cv::Sobel(img, tmp, 3, 1, 0);
	tmp = abs(tmp);
	cv::blur(tmp,tmp,cv::Size(kernel,kernel));

	for(int r =0;r<img.rows;r++){
	  for(int c =0;c<img.cols;c++){
	   if(tmp.at<short>(r,c) < thr){
	    	dispMap.at<float>(r,c) = 0;
	   }
	  }     
	}
	tmp.release();
	return dispMap;
}


cv::Mat colorFilter(cv::Mat img,cv::Mat dispMap,float thr){

  for(int r =0;r<img.rows;r++){
    for(int c =0;c<img.cols;c++){
     if(img.at<float>(r,c) > thr){
        dispMap.at<float>(r,c) = 0;
     }
    }     
  }
  return dispMap;
}



cv::Mat bilateralFilterFast(cv::Mat img){

	int kernel = 5;

	int width = img.cols;
	int height = img.rows;
	cv::Mat mask;
	cv::threshold(img,mask,1,1,cv::THRESH_BINARY);

	cv::GaussianBlur(mask,mask,cv::Size(kernel,kernel),-1);
	cv::GaussianBlur(img,img,cv::Size(kernel,kernel),-1);

	return img/mask;
}


cv::Mat noiseFilterF(cv::Mat dispMap){
	cv::Mat dispMapFinal = dispMap.clone();
	float * dispMapData = (float*)(dispMap.data);
	float * dispMapFinalData = (float*)(dispMapFinal.data);
	NoiseFilter noiseFilter(dispMapData,dispMapFinalData,dispMap.cols,dispMap.rows, 21);
	cv::parallel_for_(cv::Range(0, dispMap.rows), noiseFilter);
	dispMap.release();
	return dispMapFinal;
}

cv::Mat speckleFilterF(cv::Mat dispMap,int devider,int kernel, int percentage){

	cv::Mat dispMapFinal = dispMap.clone();

	cv::resize(dispMap,dispMap,cv::Size(dispMap.cols/devider,dispMap.rows/devider),0,0,cv::INTER_NEAREST);

	float * dispMapData = (float*)(dispMap.data);
	float * dispMapFinalData = (float*)(dispMapFinal.data);
	SpeckleFilter speckleFilter(dispMapData,dispMapFinalData,dispMap.cols,dispMap.rows, kernel/devider,(2*kernel/devider)*(2*kernel/devider)/percentage,devider);
	cv::parallel_for_(cv::Range(0, dispMap.rows), speckleFilter);
	dispMap.release();
	return dispMapFinal;
}


cv::Mat work(cv::Mat imgLeft, cv::Mat imgRight, bool sparse2, int minAreaMatchedFineTopNode2, int maxAreaMatchedFineTopNode2, int nColors2, uint kernelCostComp2, int everyNthRow2, int maxDisp2) {

	auto time0 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
	int width, height;
	width = imgLeft.cols;
	height = imgLeft.rows;

	sparse = sparse2;
	minAreaMatchedFineTopNode = minAreaMatchedFineTopNode2;
	maxAreaMatchedFineTopNode = maxAreaMatchedFineTopNode2;
	nColors = nColors2;
	kernelCostComp = kernelCostComp2;
	everyNthRow = everyNthRow2;
	maxDisp = maxDisp2;
	cv::Mat imgLeftResized;
	cv::Mat imgRightResized;

	cv::Mat sobelLeftHor;
	cv::Mat sobelRightHor;
	cv::Mat sobelLeftVert;
	cv::Mat sobelRightVert;
	cv::Mat imagesLeft;
	cv::Mat imagesRight;
	cv::Mat imgLeft2;
	cv::Mat imgRight2;


	cv::Mat sobelLeft3Hor;
	cv::Mat sobelRight3Hor;






	std::vector<cv::Mat> channelsL;
	std::vector<cv::Mat> channelsR;

	cv::Mat imL2;
	imgLeft.convertTo(imL2,CV_32F);
	cv::Mat imR2;
	imgRight.convertTo(imR2,CV_32F);

	channelsL.push_back(imL2);
	channelsR.push_back(imR2);


	cv::Sobel(imgLeft, sobelLeftHor, 5, 1, 0);
	cv::Sobel(imgRight, sobelRightHor, 5, 1, 0);

	cv::Mat slh;
	sobelLeftHor.convertTo(slh,CV_32F);
	cv::Mat srh;
	sobelRightHor.convertTo(srh,CV_32F);

	channelsL.push_back(slh);
	channelsR.push_back(srh);

	cv::Sobel(imgLeft, sobelLeftVert, 5, 0, 1);
	cv::Sobel(imgRight, sobelRightVert, 5, 0, 1);

	cv::Mat slv;
	sobelLeftVert.convertTo(slv,CV_32F);
	cv::Mat srv;
	sobelRightVert.convertTo(srv,CV_32F);

	channelsL.push_back(slv);
	channelsR.push_back(srv);

	merge(channelsL, imgLeft2);
	merge(channelsR, imgRight2);

	buildCostVolume(imgLeft2,imgRight2, maxDisp, kernelCostComp);

	cv::Mat dispMap = cv::Mat::zeros(height,width,CV_32F);

	// build tree


	MaxNode *treeLeftHor = NULL;
	MaxNode *treeRightHor = NULL;

	cv::Mat leftG;
	cv::Mat rightG;
	cv::Mat ones;

	sobelLeft3Hor = (abs(sobelLeftHor) + abs(sobelLeftVert)) / 2;
	sobelRight3Hor = (abs(sobelRightHor) + abs(sobelRightVert)) / 2;

	ones = cv::Mat::ones(cv::Size(imgLeft.cols, imgLeft.rows), CV_32F);

	sobelLeft3Hor = (255.0 * ones - sobelLeft3Hor);
	sobelRight3Hor = (255.0 * ones - sobelRight3Hor);

	sobelLeft3Hor.convertTo(leftG, CV_8U);
	sobelRight3Hor.convertTo(rightG, CV_8U);

	sobelLeft3Hor.release();
	sobelRight3Hor.release();

	leftG = leftG / (256 / nColors);
	rightG = rightG / (256 / nColors);
	leftG = leftG * (256 / nColors);
	rightG = rightG * (256 / nColors);

	leftG.convertTo(leftG, CV_32F);
	rightG.convertTo(rightG, CV_32F);
	

	treeLeftHor = buildTree(leftG);
	treeRightHor = buildTree(rightG);


	// matching

	ParallelMatching parallelMatching(dispMap, width, maxDisp, minDisp,treeLeftHor,treeRightHor);
	cv::parallel_for_(cv::Range(0, dispMap.rows*dispMap.cols), parallelMatching);

	dispMap = speckleFilterF(dispMap,2,31,16);

	 int morph_size = 1;
	  cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
	 cv::erode( dispMap, dispMap, element );

	  morph_size = 1;
	  element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

	 cv::dilate( dispMap, dispMap, element );
	 	
 	dispMap = colorFilter(imgRight, dispMap,225);
	dispMap = bilateralFilter(dispMap,8);

  return dispMap;
	
}

void setParameters() {

  minAreaMatchedFineTopNodeL = 10;
  maxAreaMatchedFineTopNodeL = 30;
  nColors = 4;
  everyNthRow = 1;

  maxDisp = 190;
  minDisp = 1;

  maxDist = 1;
  minDist = .1;

  alpha = .5;
  kernelCostCompFinalStep = 1;

  noiseFilterKernel = 7;

  thInterestingOne = 5;
  kernelCostComp = 37;
  uniquenessRatio = 1; // 12
  nThreads = 4;
}


void getParameters() {
	{
		int v;
		if(ros::param::get("~threads",v) && v != nThreads){
			ROS_INFO("Using %d threads.",v);
			nThreads = v; 
      			cv::setNumThreads(nThreads);
		}
	}
}


cv::Mat preprocess(cv::Mat img) {
  if (img.channels() > 1)
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);

  return img;
}

void inputImages_callback(const sensor_msgs::ImageConstPtr & img_msg_l,
  const sensor_msgs::ImageConstPtr & img_msg_r,
  const sensor_msgs::CameraInfoConstPtr & caminfo_msg_l) {

  ROS_INFO("Received stereo image pair");


  getParameters();


  cv::Mat img_l = cv::Mat::ones(cv::Size(img_msg_l->width, img_msg_l->height), CV_32FC3);
  cv::Mat img_r = cv::Mat::ones(cv::Size(img_msg_l->width, img_msg_l->height), CV_32FC1);

  int ctr = 0;
  for (int y = 0; y < img_msg_l->height; y++) {
    for (int x = 0; x < img_msg_l->width; x++) {
     for(int c =0;c<3;c++){
      img_l.at < cv::Vec3f > (y, x)[c] = img_msg_l->data[ctr];
      ctr++;
    }
  }
}


ctr = 0;
for (int y = 0; y < img_msg_r->height; y++) {
  for (int x = 0; x < img_msg_r->width; x++) {
   for(int c =0;c<1;c++){
    img_r.at < float > (y, x) = img_msg_r->data[ctr];   

  }
  ctr++;  

}
}






int w = img_l.cols;
int h = img_l.rows;

cv::Mat colors = img_l.clone();




img_l = preprocess(img_l);
img_r = preprocess(img_r);


cv::Mat img = work(img_l, img_r, false, minAreaMatchedFineTopNodeL, maxAreaMatchedFineTopNodeL, nColors, kernelCostComp, everyNthRow, maxDisp);




  // Publlish result
stereo_msgs::DisparityImage disp_msg;

sensor_msgs::Image msg;
msg.height = img.rows;
msg.width = img.cols;
msg.step = msg.width * sizeof(float);
msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
msg.data.resize(msg.width * msg.height * sizeof(float));
std::memcpy( & msg.data[0], & img.data[0], msg.width * msg.height * sizeof(float));
disp_msg.header.frame_id = img_msg_l->header.frame_id;
disp_msg.header.seq = img_msg_l->header.seq;
disp_msg.header.stamp.nsec = img_msg_l->header.stamp.nsec;
disp_msg.header.stamp.sec = img_msg_l->header.stamp.sec;
disp_msg.f             = caminfo_msg_l->K[0];
  disp_msg.T             = -1.f*caminfo_msg_l->P[4]; /// 0.1f;
  disp_msg.min_disparity = 0.f;
  disp_msg.max_disparity = 100000.f;
  disp_msg.delta_d       = 0.f;
  msg.header.frame_id = img_msg_l->header.frame_id;
  msg.header.seq = img_msg_l->header.seq;
  msg.header.stamp.nsec = img_msg_l->header.stamp.nsec;
  msg.header.stamp.sec = img_msg_l->header.stamp.sec;
  disp_msg.image = msg;

  chatter_pub.publish(disp_msg);

  // Publish preview

  pub_pp(img_msg_l, img, colors, caminfo_msg_l);
  colors.release();



  img = img / 5;
  std::memcpy( & msg.data[0], & img.data[0], msg.width * msg.height * sizeof(float));

  chatter_pub2.publish(msg);




  img.release();
  img_l.release();
  img_r.release();

  ROS_INFO("Published dispmap");



}


void inputImages_callback_dummy(const sensor_msgs::ImageConstPtr & img_msg_l,
  const sensor_msgs::ImageConstPtr & img_msg_r) {
  sensor_msgs::CameraInfo::Ptr dummy_ptr(new sensor_msgs::CameraInfo);
  inputImages_callback(img_msg_l, img_msg_r, dummy_ptr);
}

int main(int argc, char * argv[]) {
  ros::init(argc, argv, "MaxTreeStereoNode");
  ros::NodeHandle node;
  ROS_INFO("Registering sync'ed image streams");
  ros::TransportHints my_hints {
    ros::TransportHints().tcpNoDelay(true)
  };

  std::string leftImageTopic = "/uvc_camera/left/image_rect_color"; // uvc_cam1_rect_mono
  std::string rightImageTopic = "/uvc_camera/right/image_rect"; // uvc_cam0_rect_mono
  std::string camerainfoTopic = "/uvc_camera/cam_0/camera_info"; // uvc_cam1_rect_mono/camera_info

  setParameters();

  chatter_pub = node.advertise < stereo_msgs::DisparityImage > ("mtstereo", 5);
  chatter_pub2 = node.advertise < sensor_msgs::Image > ("mtstereo_vis", 5);

  pub_pcl = node.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("mtstereo_pcl", 5);


  bool use_caminfo_topic {
    true
  };
  if (ros::param::get("~with_caminfo_topic", use_caminfo_topic)) {
    if (use_caminfo_topic) {
      ROS_INFO("Using camera_info topic data");
    }
  }

  if (use_caminfo_topic) {
    if (ros::param::has("~image_dimensions")) {
      ROS_INFO("Parameter \"image_dimensions\" is set, but will be ignored "
        " because \"use_caminfo_topic\" is active.");
    }
    if (ros::param::has("~crop__north_east_south_west")) {
      ROS_INFO("Parameter \"crop__north_east_south_west\" is set, but will "
        "be ignored because \"use_caminfo_topic\" is active.");
    }
    std::string caminfo_topic = node.resolveName(camerainfoTopic);
    const sensor_msgs::CameraInfoConstPtr caminfo_msg_ptr =
    ros::topic::waitForMessage < sensor_msgs::CameraInfo > (caminfo_topic, node);

    image_width = caminfo_msg_ptr->width;
    image_height = caminfo_msg_ptr->height;
    crop_N = caminfo_msg_ptr->P[0];
    crop_E = caminfo_msg_ptr->P[1];
    crop_S = caminfo_msg_ptr->P[2];
    crop_W = caminfo_msg_ptr->P[3];
  }

  message_filters::Subscriber < sensor_msgs::Image > cam_l_sub(node,
    leftImageTopic, 1,
    my_hints);
  message_filters::Subscriber < sensor_msgs::Image > cam_r_sub(node,
    rightImageTopic, 1,
    my_hints);
  message_filters::Subscriber < sensor_msgs::CameraInfo > caminfo_l_sub(node,
    camerainfoTopic, 1,
    my_hints);

  std::unique_ptr < message_filters::TimeSynchronizer < sensor_msgs::Image,
  sensor_msgs::Image,
  sensor_msgs::CameraInfo >> sync_with;
  std::unique_ptr < message_filters::TimeSynchronizer < sensor_msgs::Image,
  sensor_msgs::Image >> sync_without;

  if (use_caminfo_topic) {
    sync_with.reset(new message_filters::TimeSynchronizer <
      sensor_msgs::Image,
      sensor_msgs::Image,
      sensor_msgs::CameraInfo > (cam_l_sub,
        cam_r_sub,
        caminfo_l_sub,
        1));
    sync_with->registerCallback(boost::bind( & inputImages_callback,
      _1, _2, _3));
  } else {
    sync_without.reset(new message_filters::TimeSynchronizer <
      sensor_msgs::Image,
      sensor_msgs::Image > (cam_l_sub,
        cam_r_sub,
        1));
    sync_without->registerCallback(boost::bind( & inputImages_callback_dummy,
      _1, _2));
  }

  ros::spin();

  ROS_INFO("Exiting...");
  ros::shutdown();

  return 0;
}
