/****** Dependencies ******/

#include <cmath>
#include <math.h>
#include <iostream>
#include <numeric>
#include <queue>
#include "opencv_pfm.hpp"
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
#include <vector>


/****** Definitions ******/

#define ROOT				(-1)

// Type definition of a Max-Tree node
typedef struct MaxNode	{
  int parent;
  short Area;
  bool levelroot;
  short dispL;
  short dispR;
  short dispLprev;
  short dispRprev;
  uchar nChildren;
  int begIndex;
  int matchId;
  bool curLevel;
  bool prevCurLevel;
} MaxNode;

// Type definition of matching of two nodes
typedef struct Match	{
  float cost;
  int match;
} Match;

typedef cv::Vec<float,2> Vec2f;


/****** Parameters ******/

float alpha = .8; // alpha
int minAreaMatchedFineTopNode = 3; // omega_alpha
float factorMax = 3; // omega_beta = image_width/factorMax
int nColors = 5; // q
int sizes[16] = {1,0,0,0}; // S, appended with 0,0 which correspond to the refinement step
int total = 4; // #S + 2
uint kernelCostComp = 6; // omega_delta
bool sparse = true; // produce sparse (true) or semi-dense (false) disparity map


/****** Global variables ******/

int maxAreaMatchedFineTopNode;
float weightContext = 1-alpha;
const char * outFolderGV;
int skipY = 1;
bool removeLongOnes = false;
float step = 1;
bool useLRCHECK = true;
bool useBlurOnG = true;
bool skipsides = true;
bool Verydense = false;
int disparityLevelsResized;
const char* filenameOutput;
int disparityLevels;


/****** Functions ******/

/**
 *
 * Get the current time
 *
 * @return  	The current time in seconds
 *
 */
double getWallTime(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

/**
 *
 * Clear (i.e. deallocate the memory) of a 2D vector
 *
 * @param    std::vector<std::vector<int>> vect  The vector to clear.
 *
 */
void clear2Dvect(std::vector<std::vector<int>> vect) {
	for (uint i = 0; i < vect.size(); i++) {
		vect[i].clear();
	}
	vect.clear();
}





void writeVisDispmap(cv::Mat img, std::string str, float dispLevels) {

dispLevels = 70;

  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      img.at < float > (r, c) = img.at < float > (r, c) > 100000 ? 0 : img.at < float > (r, c);
    }
  }
  img = img * 255.0 / dispLevels;

  img.convertTo(img, CV_8U);
  cv::applyColorMap(img, img, cv::COLORMAP_JET);
  cv::imwrite(str.c_str(), img);

}


void writeLine(std::string filename,std::string line){
	std::ofstream ofs;
	ofs.open (filename, std::ofstream::out | std::ofstream::app);

	ofs << line << std::endl;

	ofs.close();
}
void clearFile(std::string filename){
	std::ofstream ofs;
	ofs.open(filename, std::ofstream::out | std::ofstream::trunc);
	ofs.close();
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
std::vector<int> getNeighborsBottom(MaxNode * tree, int indexCenter, int width,
		int height, bool useCur, bool dispCheck,int kernelCostComp) {
	std::vector<int> tmp;

	int prevIndex = indexCenter;

	if (!dispCheck
			|| (tree[prevIndex].dispLprev >= 0 && tree[prevIndex].dispRprev >= 0))
		tmp.push_back(prevIndex);

	for (int row = 0; tmp.empty() || tmp.size() < kernelCostComp; row++) {
		int index = tree[prevIndex].begIndex + tree[prevIndex].Area / 2;

		index = index - skipY*width;

		if (index < 0 || index >= width * height)
			break;

		if (useCur) {
			while (index != ROOT && !tree[index].curLevel) {
				index = tree[index].parent;
			}
		} else {
			while (index != ROOT && !tree[index].prevCurLevel) {
				index = tree[index].parent;
			}
		}

		if (index != ROOT) {
			if (!dispCheck
					|| (tree[index].dispLprev >= 0 && tree[index].dispRprev >= 0))
				tmp.push_back(index);
		} else {
			break;
		}

		prevIndex = index;
	}

	return tmp;
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
std::vector<int> getNeighborsTop(MaxNode * tree, int indexCenter, int width,
		int height, bool useCur, bool dispCheck,int kernelCostComp) {
	std::vector<int> tmp;

	int prevIndex = indexCenter;

	if (!dispCheck
			|| (tree[prevIndex].dispLprev >= 0 && tree[prevIndex].dispRprev >= 0))
		tmp.push_back(prevIndex);

	for (int row = 0; tmp.empty() || tmp.size() < kernelCostComp; row++) {
		int index = tree[prevIndex].begIndex + tree[prevIndex].Area / 2;

		index = index + skipY*width;

		if (index < 0 || index >= width * height)
			break;

		if (useCur) {
			while (index != ROOT && !tree[index].curLevel) {
				index = tree[index].parent;
			}
		} else {
			while (index != ROOT && !tree[index].prevCurLevel) {
				index = tree[index].parent;
			}
		}

		if (index != ROOT) {
			if (!dispCheck
					|| (tree[index].dispLprev >= 0 && tree[index].dispRprev >= 0))
				tmp.push_back(index);
		} else {
			break;
		}

		prevIndex = index;
	}

	return tmp;
}

//
/**
 *
 * Calculate the disparity search range for a node with index
 *
 * @param    MaxNode * tree Max-Tree
 * @param    int index Index of the node
 * @param    int * res Stores the result: the disparity search range for a node with index.
 * @param    int dispLevels The maximum disparity offset allowed.
 * @param	int width Width of image
 * @param	int height Height of image
 * @param   int kernel Number of neighbors above and below the node considered in the calculation.
 *
 */
void calcDispSearchRange(MaxNode * tree, int index, int * res, int dispLevels, int width,
		int height,int kernel) {
	res[0] = 0;
	res[1] = dispLevels;

	std::vector<int> neighTop;
	std::vector<int> neighBottom;

	if (index < 0 || index >= width * height) {

		return;
	} else {
		neighTop = getNeighborsTop(tree, index, width, height, false, true,kernel);
		neighBottom = getNeighborsBottom(tree, index, width, height, false,
				true,kernel);
	}

	std::vector<int> L;
	std::vector<int> R;

	for (uint i = 0; i < neighTop.size(); i++) {
		if (neighTop[i] < 0 || neighTop[i] >= width * height)
			continue;

		L.push_back(tree[neighTop[i]].dispLprev);
		R.push_back(tree[neighTop[i]].dispRprev);

	}
	for (uint i = 0; i < neighBottom.size(); i++) {
		if (neighBottom[i] < 0 || neighBottom[i] >= width * height)
			continue;

		L.push_back(tree[neighBottom[i]].dispLprev);
		R.push_back(tree[neighBottom[i]].dispRprev);
	}

	if (L.size() >= 1 && R.size() >= 1) {
		sort(L.begin(), L.end());
		sort(R.begin(), R.end());

		int medianL = L[L.size() / 2];
		int medianR = R[R.size() / 2];

		res[1] = medianR;
		res[0] = medianL;
	}

	L.clear();
	R.clear();

	neighTop.clear();
	neighBottom.clear();
}


/**
 *
 * Compute the gradient cost between a node pair
 *
 * @param    cv::Mat imgL The left images
 * @param    cv::Mat imgR The right images
 * @param    int type Type of images
 * @param    MaxNode* treeLeft Left Max-tree
 * @param    MaxNode* treeRight Right Max-tree
 * @param    int indexL Index of the left node
 * @param     int indexR Index of the right node
 * @return   Gradient cost
 *
 */
float gradientCostComp(cv::Mat imgL, cv::Mat imgR, int type, MaxNode* treeLeft,
		MaxNode* treeRight, int indexL, int indexR) {

	int width = imgL.cols;
	int height = imgL.rows;

	int areaL = treeLeft[indexL].Area - 1;
	int areaR = treeRight[indexR].Area - 1;

	int RightL = (treeLeft[indexL].begIndex + areaL) % width  ;
	int LeftL = treeLeft[indexL].begIndex % width ;
	int RightR = (treeRight[indexR].begIndex  + areaR) % width  ;
	int LeftR = treeRight[indexR].begIndex  % width ;

	int y = imgL.rows - 1 - (treeRight[indexR].begIndex / width);

	if (!(y >= 0 && y < height)) {
		return std::numeric_limits<int>::max();
	}

	float costL = 0;
	float costR = 0;

	for (float ratio = 0; ratio <= 1; ratio += step) {
		float tmpLeft = LeftL * ratio + (1 - ratio) * RightL;
		float tmpRight = LeftR * ratio + (1 - ratio) * RightR;
		float tmp= imgL.at<Vec2f>(y, tmpLeft)[0]-imgR.at<Vec2f>(y, tmpRight)[0];
		costL += tmp > 0? tmp:-tmp;
		tmp = imgL.at<Vec2f>(y, tmpLeft)[1]- imgR.at<Vec2f>(y, tmpRight)[1];
		costR +=  tmp > 0? tmp:-tmp;
	}

	return (costL + costR);
}

/**
 *
 * Compute the context cost between a node pair
 *
 * @param    int indexL Index of the left node
 * @param    int indexR Index of the right node
 * @param    MaxNode* treeLeft Left Max-tree
 * @param    MaxNode* treeRight Right Max-tree
 * @return   Context cost
 *
 */
int contextCost(int indexL, int indexR, MaxNode* treeHorL, MaxNode* treeHorR) {

	if(indexL >= 0 && indexR >= 0){
		float total = 0;

		float sum = 0;

		float weight = 1;

		while (indexL != ROOT && indexR != ROOT) {

			sum += weight * abs(((float)treeHorL[indexL].Area / (float)(treeHorL[indexL].Area+treeHorR[indexR].Area ))-.5 );

			indexL = treeHorL[indexL].parent;
			indexR = treeHorR[indexR].parent;

			total += weight;
			//weight/=2;
		}
		return (225)*(sum / total);
	}else{
		return std::numeric_limits<int>::max();
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
void getTops(MaxNode * treeRightHor, MaxNode * treeLeftHor, int width,
		int height, std::vector<std::vector<int>> *topsL,
		std::vector<std::vector<int>> * topsR) {

	for (int r = 0; r < height; r++) {
		std::vector<int> rowL;
		std::vector<int> rowR;

		for (int c = 0; c < width; c++) {
			int index = c + (height - 1 - r) * width;
			if (treeRightHor[index].levelroot
					&& treeRightHor[index].nChildren == 0) {
				if ((treeRightHor[index].Area > minAreaMatchedFineTopNode) && treeRightHor[index].Area  < maxAreaMatchedFineTopNode)
					rowR.push_back(index);
			}
			if (treeLeftHor[index].levelroot
					&& treeLeftHor[index].nChildren == 0) {
				if ((treeLeftHor[index].Area > minAreaMatchedFineTopNode) && treeLeftHor[index].Area  < maxAreaMatchedFineTopNode)
					rowL.push_back(index);
			}
		}

		(*topsL).push_back(rowL);
		(*topsR).push_back(rowR);
	}
}

/**
 *
 * Clear markings of nodes as prevCurLevel
 *
 * @param    int width Width of the input image
 * @param    int height Height of the input image
 * @param    MaxNode* tree Max-Tree
 *
 */
void clearPrevCurLevel(int width, int height, MaxNode* tree) {
	for (int i = 0; i < width * height; i++) {
		tree[i].prevCurLevel = false;
	}
}

/**
 *
 * Clear markings of nodes as curLevel
 *
 * @param    int width Width of the input image
 * @param    int height Height of the input image
 * @param    MaxNode* tree Max-Tree
 *
 */
void clearCurLevel(int width, int height, MaxNode* tree) {
	for (int i = 0; i < width * height; i++) {
		tree[i].prevCurLevel = tree[i].curLevel;
		tree[i].curLevel = false;
	}
}





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
	
	if(n == 0){
 		for (int r = 0; r < height; r++){
			std::vector<int> LevelNChildrenL;
			for (uint c = 0; c < topsL[r].size(); c++) {
				int current = topsL[r][c];
				while (current != ROOT && !treeLeftHor[current].levelroot) {
					current = treeLeftHor[current].parent;
				}
				if (current != ROOT){
					LevelNChildrenL.push_back(current);
					treeLeftHor[current].curLevel=true;
				}
				
			}
			std::vector<int> LevelNChildrenR;
			for (uint c = 0; c < topsR[r].size(); c++) {
				int current = topsR[r][c];
				while (current != ROOT && !treeRightHor[current].levelroot) {
					current = treeRightHor[current].parent;
				}
				if (current != ROOT){
					LevelNChildrenR.push_back(current);
					treeRightHor[current].curLevel=true;
				}
				
			}
			topsLnTh->at(r) = LevelNChildrenL;
			topsRnTh->at(r) = LevelNChildrenR;
		}
	

	}else{
		NthChildrenSide nthChildrenSide(topsL, n, topsLnTh, treeLeftHor);
		cv::parallel_for_(cv::Range(0, height), nthChildrenSide);

		NthChildrenSide nthChildrenSide2(topsR, n, topsRnTh, treeRightHor);
		cv::parallel_for_(cv::Range(0, height), nthChildrenSide2);
	}



}

class MatchTopsParallelbody: public cv::ParallelLoopBody {

private:
	MaxNode *treeRightHor;
	MaxNode *treeLeftHor;
	int disparityLevels;
	const cv::Mat imgRight;
	const cv::Mat imgLeft;
	std::vector<std::vector<int>> topsL;
	std::vector<std::vector<int>> topsR;
	int iteration;
	int width;
	int height;
	cv::Mat * dispMap;

public:
	MatchTopsParallelbody(MaxNode *treeRightHor, MaxNode *treeLeftHor,
			int disparityLevels, const cv::Mat imgRight, const cv::Mat imgLeft,
			std::vector<std::vector<int>> topsL,
			std::vector<std::vector<int>> topsR, int iteration, int width,
			int height, cv::Mat * dispMap) :
			treeRightHor(treeRightHor), treeLeftHor(treeLeftHor), disparityLevels(
					disparityLevels), imgRight(imgRight), imgLeft(imgLeft), topsL(
					topsL), topsR(topsR), iteration(iteration), width(width), height(
					height), dispMap(dispMap) {
	}

	virtual void operator()(const cv::Range& range) const {
		for (int r = range.start; r < range.end; r++) {

			std::vector<Match> matchesRowL(topsL[r].size());
			std::vector<Match> matchesRowR(topsR[r].size());

			for (uint i = 0; i < matchesRowL.size(); i++) {
				matchesRowL[i].cost = std::numeric_limits<float>::infinity();
			}
			for (uint i = 0; i < matchesRowR.size(); i++) {
				matchesRowR[i].cost = std::numeric_limits<float>::infinity();
			}

			// Calculate matching costs for row
			for (uint LvInd = 0; LvInd < topsL[r].size(); LvInd++) {
				int c2 = treeLeftHor[topsL[r][LvInd]].begIndex % width;
				int c2_end = treeLeftHor[topsL[r][LvInd]].begIndex % width
						+ treeLeftHor[topsL[r][LvInd]].Area - 1;

				int index = topsL[r][LvInd];

				if (0 != iteration) {

					while (index != ROOT && !treeLeftHor[index].prevCurLevel) {
						index = treeLeftHor[index].parent;
					}
					if (index == ROOT)
						continue;
				}

				int c2_parent = treeLeftHor[index].begIndex % width;
				int c2_end_parent = treeLeftHor[index].begIndex % width
						+ treeLeftHor[index].Area - 1;

				int fromBeg =
						c2 - disparityLevels >= 0 ? c2 - disparityLevels : 0;
				int toEnd = c2_end >= 0 ? c2_end : 0;

				int begBeg, endEnd;



				if (0 != iteration) {

					int * res = (int *) malloc(2 * sizeof(int));

					calcDispSearchRange(treeLeftHor, index, res, disparityLevels, width, height,kernelCostComp); // iteration==total-1?0:kernelCostComp

					int n0 = res[0];
					int n1 = res[1];

					begBeg = c2_parent-n0 >=0?c2_parent-n0:0;
					endEnd = c2_end_parent-n1 >=0?c2_end_parent-n1:0;

					if ( total - 2 == iteration) {

						if ((!skipsides || (c2_end != width - 1
														&& c2 != 0))) {
						} else {
							treeLeftHor[index].dispL = -1;
							treeLeftHor[index].dispR = -1;
						}

					}

					if (total - 1 == iteration) {

						float min =  res[1] > res[0]? res[0]:res[1];
						float max =  res[1] < res[0]? res[0]:res[1];

						if (max < disparityLevelsResized  && min >= 0 && (!skipsides
												|| (c2_end != width - 1
														&& c2 != 0))) {

							if(sparse){
								dispMap->at<float>(r, c2) = res[0]; //res[0] == min ? min : -1;min;
								dispMap->at<float>(r, c2_end) = res[1]; //res[1] == min ? min : -1; min;
							}else{
								for (int column = c2; column <= c2_end; column++) {
									dispMap->at<float>(r, column) = min;
								}
							}

						} else {
							for (int column = c2; column <= c2_end; column++) {
								dispMap->at<float>(r, column) = -1;
							}
						}
					}

					free(res);
				} else {
					begBeg = fromBeg;
					endEnd = toEnd;
				}


				if (total - 2 <= iteration)
						continue;

				for (uint RvInd = 0;  RvInd < topsR[r].size(); RvInd++) {


					int c = treeRightHor[topsR[r][RvInd]].begIndex % width;
					int c_end = treeRightHor[topsR[r][RvInd]].begIndex % width
							+ treeRightHor[topsR[r][RvInd]].Area - 1;

					if (c <= c2 && c_end <= c2_end
							&& (begBeg <= c && c <= endEnd)
							&& (begBeg <= c_end && c_end <= endEnd)) { //

						float cost = 0;

						int indexL = topsL[r][LvInd];
						int indexR = topsR[r][RvInd];



						std::vector<int> NeighborsLTop = getNeighborsTop(
								treeLeftHor, indexL, width, height, true,
								false,kernelCostComp);
						std::vector<int> NeighborsRTop = getNeighborsTop(
								treeRightHor, indexR, width, height, true,
								false,kernelCostComp);
						std::vector<int> NeighborsLBottom =
								getNeighborsBottom(treeLeftHor, indexL, width,
										height, true, false,kernelCostComp);
						std::vector<int> NeighborsRBottom =
								getNeighborsBottom(treeRightHor, indexR, width,
										height, true, false,kernelCostComp);

						float colCost = 0;
						float descendentsCost = 0;
						float sizeTop =
								NeighborsLTop.size() > NeighborsRTop.size() ?
										NeighborsRTop.size() :
										NeighborsLTop.size();
						float sizeBottom =
								NeighborsLBottom.size()
										> NeighborsRBottom.size() ?
										NeighborsRBottom.size() :
										NeighborsLBottom.size();


						for (int i = 0; i < sizeTop; i++) {

							float tmp = gradientCostComp(imgLeft, imgRight, 0,
									treeLeftHor, treeRightHor,
									NeighborsLTop[i], NeighborsRTop[i]);
							colCost += tmp;

							tmp = contextCost(NeighborsLTop[i],NeighborsRTop[i],treeLeftHor,treeRightHor);
							descendentsCost += tmp;
						}

						for (int i = 0; i < sizeBottom; i++) {
							float tmp = gradientCostComp(imgLeft, imgRight, 0,
									treeLeftHor, treeRightHor,
									NeighborsLBottom[i], NeighborsRBottom[i]);
							colCost += tmp;

							tmp = contextCost(NeighborsLBottom[i],NeighborsRBottom[i],treeLeftHor,treeRightHor);
							descendentsCost += tmp;
						}

						cost += (colCost / (sizeBottom + sizeTop - 1))
								* alpha;
						cost += (descendentsCost / (sizeBottom + sizeTop - 1))
								* weightContext;

						if (matchesRowL[LvInd].cost > cost) {
							matchesRowL[LvInd].cost = cost;
							matchesRowL[LvInd].match = RvInd;
						};
						if (matchesRowR[RvInd].cost > cost) {
							matchesRowR[RvInd].cost = cost;
							matchesRowR[RvInd].match = LvInd;
						}

						NeighborsLTop.clear();
						NeighborsRTop.clear();
						NeighborsLBottom.clear();
						NeighborsRBottom.clear();

					} else {
						if (c > endEnd)
							break;
					}
				}
			}

			if (total - 2 <= iteration)
					continue;

			// Left-right consistency check
			for (uint LvInd = 0; LvInd < topsL[r].size(); LvInd++) {

				if (matchesRowL[LvInd].cost
						== std::numeric_limits<float>::infinity()) {
					continue;
				}

				Match L = matchesRowL[LvInd];

				if (matchesRowR[L.match].cost
						== std::numeric_limits<float>::infinity())
					continue;

				Match R = matchesRowR[L.match];

				int tmL = L.match;
				int tmR = LvInd;


				if ((R.match == tmR || !useLRCHECK)) {

					int cLeft = treeRightHor[topsR[r][tmL]].begIndex % width;
					int c2Left = treeLeftHor[topsL[r][tmR]].begIndex % width;
					int cRight = treeRightHor[topsR[r][tmL]].begIndex % width
							+ treeRightHor[topsR[r][tmL]].Area - 1;
					int c2Right = treeLeftHor[topsL[r][tmR]].begIndex % width
							+ treeLeftHor[topsL[r][tmR]].Area - 1;

					if ((cLeft == 0 || c2Left == 0 || cRight == width - 1
							|| c2Right == width - 1)) {
						treeLeftHor[topsL[r][LvInd]].dispL = -1;
						treeLeftHor[topsL[r][LvInd]].dispR = -1;
						treeLeftHor[topsL[r][LvInd]].matchId = -1;

						continue;
					}

					int dispLeft = abs(c2Left - cLeft);
					int dispRight = abs(c2Right - cRight);

					// custom border cases:

					int assignedRight =
							cRight >= width -1 || c2Right >= width -1  ?
									dispLeft : dispRight;
					int assignedLeft =
							cLeft <= 0 || c2Left <= 0 ? dispRight : dispLeft;

					treeRightHor[topsR[r][tmL]].dispL = assignedLeft;
					treeRightHor[topsR[r][tmL]].dispR = assignedRight;

					treeLeftHor[topsL[r][tmR]].dispL = assignedLeft;
					treeLeftHor[topsL[r][tmR]].dispR = assignedRight;

					treeRightHor[topsR[r][tmL]].matchId = topsL[r][tmR];
					treeLeftHor[topsL[r][tmR]].matchId = topsR[r][tmL];

				} else {
					treeLeftHor[topsL[r][LvInd]].dispL = -1;
					treeLeftHor[topsL[r][LvInd]].dispR = -1;
					treeLeftHor[topsL[r][LvInd]].matchId = -1;
				}
			}
		}
	}
};


/**
 *
 * match coarse top nodes of the same level
 *
 * @param MaxNode *treeRightHor The right Max-Tree
 * @param  MaxNode *treeLeftHor  The left Max-Tree
 * @param  int disparityLevels The maximum disparity
 * @param  const cv::Mat imgRight the right images
 * @param  const cv::Mat imgLeft the left images
 * @param  std::vector<std::vector<int>> topsL The coarse top nodes of the left image
 * @param  std::vector<std::vector<int>> topsR The coarse top nodes of the right image
 * @param  int iteration The level
 * @param  cv::Mat * dispMap The resulting disparity map
 *
 */
void matchNodes(MaxNode *treeRightHor, MaxNode *treeLeftHor,
		int disparityLevels, const cv::Mat imgRight, const cv::Mat imgLeft,
		std::vector<std::vector<int>> topsL,
		std::vector<std::vector<int>> topsR, int iteration, cv::Mat * dispMap) {

	int width = imgLeft.cols;
	int height = imgLeft.rows;

	parallel_for_(cv::Range(0, height),
			MatchTopsParallelbody(treeRightHor, treeLeftHor, disparityLevels,
					imgRight, imgLeft, topsL, topsR, iteration, width, height,
					dispMap));
}





class BuildTreeParallel: public cv::ParallelLoopBody {

private:
	int height;
	int width;
	cv::Mat gval;
	MaxNode* node;
public:

	BuildTreeParallel(int height, int width,cv::Mat gval,MaxNode * node) : height(height), width(width), gval(gval),node(node) {
	}

	virtual void operator()(const cv::Range& range) const {
		for (int row = range.start; row < range.end; row++) {
			int current,i,next,curr_parent;
			int invRow = height -1 - row;

			for (i = 0, current = row * width; i < width; i++, current++){
					node[current].parent = ROOT;
				//	node[current].gval = gval.at<uchar>(invRow,current % width);
					node[current].Area = 1;
					node[current].levelroot = true;
					node[current].nChildren = 0;
					node[current].begIndex = std::numeric_limits<int>::max();
					node[current].dispL = -1;
					node[current].dispR = -1;
					node[current].matchId = -1;
					node[current].curLevel = false;
					node[current].prevCurLevel = false;
				}

				current = row * width; // reset the counter

				for (i = 1, next = current + 1; i < width; i++, next++){
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
					if (gval.at<uchar>(invRow,current % width) <= gval.at<uchar>(invRow,next % width)) { /* ascending or flat */

						node[next].parent = current;

						if (gval.at<uchar>(invRow,current % width) == gval.at<uchar>(invRow,next % width)){ /*flat */
							node[current].Area ++;
							node[next].levelroot = false;
						} else {
							current = next;  /* new top level root */
						}
					} else { /* descending */
						curr_parent = node[current].parent; // save the current node parent

						/*
						 * For each parent of the current node, until a root node is reached or
						 * when the gray value of the current node becomes lower, this procedure
						 * of swapping parents should be repeated.
						 */
						while ((curr_parent!=ROOT) && (gval.at<uchar>(invRow,curr_parent % width) >gval.at<uchar>(invRow,next % width) )){
							node[curr_parent].Area += node[current].Area;
							current = curr_parent;
							curr_parent = node[current].parent;
						}
						node[current].parent = next;
						if(gval.at<uchar>(invRow,current % width) == gval.at<uchar>(invRow,node[current].parent % width)  && next != ROOT)
							node[current].levelroot = false;
						node[next].Area += node[current].Area;
						node[next].parent = curr_parent;
						if(gval.at<uchar>(invRow,node[next].parent % width) == gval.at<uchar>(invRow,next % width) && curr_parent != ROOT)
							node[next].levelroot = false;
						current = next;
					}
				}

				/*
				 * Go through the root path to update the area value of the root node.
				 */
				curr_parent = node[current].parent;
				while (curr_parent != ROOT){
					node[curr_parent].Area += node[current].Area;
					current = curr_parent;
					curr_parent = node[current].parent;
				}

				for (i = row * width; i < (row+1) * width; i++){
					int current = i;
					if(node[current].levelroot && node[current].parent != ROOT){
						while(!node[node[current].parent].levelroot){
							current = node[current].parent;
						}
						node[node[current].parent].nChildren++;
					}
				}

				for (i = row * width; i < (row+1) * width; i++){
					int current = i;
					if(node[current].nChildren == 0 && node[current].levelroot ){
						int smallest = current;
						while(node[current].parent != ROOT){
							if(current < smallest){
								smallest = current;
							}
							if(node[current].begIndex > smallest){
								node[current].begIndex = smallest;
							}
							current = node[current].parent;
						}
					}
				}
			}
	}
};


class NoiseFilter : public cv::ParallelLoopBody{
public:
  NoiseFilter (short * dispMap,short * dispMapFinal,int cols,int rows, const int kernel)
  :  dispMap(dispMap),dispMapFinal(dispMapFinal),cols(cols),rows(rows), kernel(kernel){}
  virtual void operator ()(const cv::Range& range) const {
	for (int r = range.start; r < range.end; r++) {
	    for (int c = 0; c < cols; c++) {
	
		float curval = dispMap[cols * (r) + (c)];
		if (curval > 1) {
			int total = 0;
			int totalMean = 0;
			int rBeg = r - kernel > 0 ? r - kernel : r;
			int cBeg = c - kernel > 0 ? c - kernel : c;
			int cEnd = (c + kernel <= cols ? c + kernel : c);
			int rEnd = (r + kernel <= rows ? r + kernel : r);
		  	for (int cOff = cBeg; cOff < cEnd; cOff++) {
				short absCoff = abs(cOff-c);
				for (int rOff = rBeg; rOff < rEnd; rOff++) {
					short val = dispMap[cols * (rOff) + (cOff)];
					if (val > 1) {
					    if (abs(val - curval) <= 3) {
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
	short * dispMap;
	short * dispMapFinal;
	int cols;
	int rows;
	const int kernel;
};


/**
 *
 * Build a max tree from an image
 *
 * @param    cv::Mat gval The image
 * @return   The generated Max-Tree
 *
 */
MaxNode* buildTree(cv::Mat gval){
	int height = gval.rows;
	int width = gval.cols;
	MaxNode* node = (MaxNode*)malloc(width*height*sizeof(MaxNode));

	parallel_for_(cv::Range(0, height), BuildTreeParallel(height, width, gval,node));

	return node;
}


cv::Mat work(cv::Mat imgLeft, cv::Mat imgRight,bool sparse2,float alpha2,int minAreaMatchedFineTopNode2,int maxAreaMatchedFineTopNode2,int nColors2,int * sizes2, int total2,uint kernelCostComp2,int kernelCostVolume, int everyNthRow2, int disparityLevels2,float minConfidencePercentage2, float allowanceSearchRange2){
	
	sparse = sparse2;

	disparityLevels = disparityLevels2;
	cv::Mat imgLeftResized;
	cv::Mat imgRightResized;

	cv::Mat ones;

	maxAreaMatchedFineTopNode = maxAreaMatchedFineTopNode2;

	// resize left & right images

	disparityLevelsResized = disparityLevels;

	int width, height;
	
	// read left & right images

	width = imgLeft.cols;
	height = imgLeft.rows;

	// resize left & right images


	// convert left & right images to grayscale

	double startTotal = getWallTime();

	cv::medianBlur(imgLeft,imgLeft,5);
	cv::medianBlur(imgRight,imgRight,5);

	// horizontal Sobel

	MaxNode *treeLeftHor = NULL;
	MaxNode *treeRightHor = NULL;
	cv::Mat sobelLeft3Hor;
	cv::Mat sobelRight3Hor;
	cv::Mat sobelLeftHor;
	cv::Mat sobelRightHor;
	cv::Mat sobelLeft3ucharHor;
	cv::Mat sobelRight3ucharHor;

	cv::Sobel(imgLeft, sobelLeftHor, 5, 1, 0);
	cv::Sobel(imgRight, sobelRightHor, 5, 1, 0);

	// vertical Sobel

	cv::Mat sobelLeft3Vert;
	cv::Mat sobelRight3Vert;
	cv::Mat sobelLeftVert;
	cv::Mat sobelRightVert;
	cv::Mat sobelLeft3ucharVert;
	cv::Mat sobelRight3ucharVert;

	cv::Sobel(imgLeft, sobelLeftVert, 5, 0, 1);
	cv::Sobel(imgRight, sobelRightVert, 5, 0, 1);

	// compute G

	sobelLeft3Hor = (abs(sobelLeftHor) + abs(sobelLeftVert)) / 2;
	sobelRight3Hor = (abs(sobelRightHor) + abs(sobelRightVert)) / 2;

	ones = cv::Mat::ones(cv::Size(imgLeft.cols, imgLeft.rows), CV_32F);

	sobelLeft3Hor = (255.0 * ones - sobelLeft3Hor);
	sobelRight3Hor = (255.0 * ones - sobelRight3Hor);

	float fmin = 125;
	sobelLeft3Hor = ((sobelLeft3Hor - fmin)/(255-fmin))*255;
	sobelRight3Hor  = ((sobelRight3Hor - fmin)/(255-fmin))*255;

	sobelLeft3Hor.convertTo(sobelLeft3ucharHor, CV_8U);
	sobelRight3Hor.convertTo(sobelRight3ucharHor, CV_8U);

	sobelLeft3Hor.release();
	sobelRight3Hor.release();

	sobelLeft3ucharHor = sobelLeft3ucharHor / (256 / nColors);
	sobelRight3ucharHor = sobelRight3ucharHor / (256 / nColors);
	sobelLeft3ucharHor = sobelLeft3ucharHor * (256 / nColors);
	sobelRight3ucharHor = sobelRight3ucharHor * (256 / nColors);

	// merge into 3D matrix

	std::vector<cv::Mat> imgLeftChannelsL;
	imgLeftChannelsL.push_back(sobelLeftHor);
	imgLeftChannelsL.push_back(sobelLeftVert);
	cv::Mat imgLeft2;
	cv::merge(imgLeftChannelsL, imgLeft2);

	std::vector<cv::Mat> imgRightChannelsL;
	imgRightChannelsL.push_back(sobelRightHor);
	imgRightChannelsL.push_back(sobelRightVert);
	cv::Mat imgRight2;
	cv::merge(imgRightChannelsL, imgRight2);

	// create max trees

	treeLeftHor = buildTree(sobelLeft3ucharHor);
	treeRightHor = buildTree(sobelRight3ucharHor);

	sobelLeft3ucharHor.release();
	sobelRight3ucharHor.release();
	sobelLeft3ucharVert.release();
	sobelRight3ucharVert.release();

	// Get tops

	std::vector<std::vector<int>> topsL;
	std::vector<std::vector<int>> topsR;

	std::vector<std::vector<int>> topsLnTh(height);
	std::vector<std::vector<int>> topsRnTh(height);
	std::vector<std::vector<int>> topsLnThPrev(height);
	std::vector<std::vector<int>> topsRnThPrev(height);

	getTops(treeRightHor, treeLeftHor, width, height, &topsL, &topsR);

	// initialize dispmap with -1

	cv::Mat dispMap;
	dispMap = cv::Mat::ones(cv::Size(imgLeft.cols, imgLeft.rows), CV_32F);
	dispMap *= -1;

	// coarse to fine levels

	for (int i = 0; i < total; i++) {

		if(i == total - 2)
			continue;

		for(int ctr = 0;ctr< width*height;ctr++){
			treeLeftHor[ctr].dispLprev =  treeLeftHor[ctr].dispL;
			treeLeftHor[ctr].dispRprev =  treeLeftHor[ctr].dispR;
			treeRightHor[ctr].dispLprev =  treeRightHor[ctr].dispL;
			treeRightHor[ctr].dispRprev =  treeRightHor[ctr].dispR;
		}
		
		clearPrevCurLevel(width, height, treeLeftHor);
		clearPrevCurLevel(width, height, treeRightHor);
		clearCurLevel(width, height, treeLeftHor);
		clearCurLevel(width, height, treeRightHor);

		if(i < 1 || sizes[i-1] != 0){
			getNthTops(topsR, topsL, sizes[i], &topsRnTh, &topsLnTh, treeRightHor,treeLeftHor, height);
		}


		matchNodes(treeRightHor, treeLeftHor, disparityLevels, imgRight2, imgLeft2, topsLnTh, topsRnTh, i, &dispMap);
	
		if(sizes[i] == 1){
		
			if (i > 0) {
				clear2Dvect(topsLnThPrev);
				clear2Dvect(topsRnThPrev);
			}

			topsLnThPrev = topsLnTh;
			topsRnThPrev = topsRnTh;

			topsLnTh = std::vector<std::vector<int>>(height);
			topsRnTh = std::vector<std::vector<int>>(height);
		}
	}


	if (sparse) {
		cv::Mat dispMapFinal;
		cv::Mat dispMapShort;
		dispMap.convertTo(dispMapFinal, CV_16S);
		dispMap.convertTo(dispMapShort, CV_16S);

		short *dispMapData = (short*)(dispMapShort.data);
		short *dispMapFinalData = (short*)(dispMapFinal.data);

		NoiseFilter noiseFilter2(dispMapData,dispMapFinalData,dispMapShort.cols,dispMapShort.rows, 21);
		cv::parallel_for_(cv::Range(0, dispMapShort.rows), noiseFilter2);
		dispMap.release();

		dispMapFinal.convertTo(dispMap, CV_32F);
	}

	// Free memory

	ones.release();
	imgLeft.release();
	imgRight.release();
	imgLeft2.release();
	imgRight2.release();
	sobelRightVert.release();
	sobelLeftVert.release();
	sobelRightHor.release();
	sobelLeftHor.release();
	free(treeLeftHor);
	free(treeRightHor);

	dispMap.convertTo(dispMap, CV_32F);

	return dispMap;

}
