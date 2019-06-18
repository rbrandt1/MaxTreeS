/****** Load Dependencies ******/

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

typedef struct MaxNode	{
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
} MaxNode;

typedef struct Match	{
  float cost;
  int match;
} Match;

typedef cv::Vec<float,2> Vec2f;


/****** Parameters ******/

float alpha = .8; // alpha
int minAreaMatchedFineTopNode = 0; // omega_alpha
float factorMax = 3; // omega_beta = image_width/factorMax
int nColors = 5; // q
int sizes[16] = {1,0,0,0}; // S, appended with 0,0
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

double getWallTime(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void clear2Dvect(std::vector<std::vector<int>> vect) {
	for (uint i = 0; i < vect.size(); i++) {
		vect[i].clear();
	}
	vect.clear();
}

std::vector<int> getNeighboursBottom(MaxNode * tree, int indexCenter, int width,
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

std::vector<int> getNeighboursTop(MaxNode * tree, int indexCenter, int width,
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

void calcCI(MaxNode * tree, int index, int * res, int dispLevels, int width,
		int height,int kernel) {
	res[0] = 0;
	res[1] = dispLevels;

	std::vector<int> neighTop;
	std::vector<int> neighBottom;

	if (index < 0 || index >= width * height) {

		return;
	} else {
		neighTop = getNeighboursTop(tree, index, width, height, false, true,kernel);
		neighBottom = getNeighboursBottom(tree, index, width, height, false,
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

void plotDispmap(cv::Mat dispMap, int disparityLevelsResized, bool side) {

	for (int r = 0; r < dispMap.rows; r++) {
		for (int c = 0; c < dispMap.cols; c++) {
			if (dispMap.at<float>(r, c) < 0) {
				dispMap.at<float>(r, c) =
						std::numeric_limits<float>::infinity();
			}
		}
	}

	std::string str = "";
	str.append(outFolderGV);
	str.append("/");
	str.append(filenameOutput);
	str.append(".pfm");
	opencv_pfm::imwrite_pfm(str.c_str(), dispMap,1,1);
}

/* cost computation */

float costComp(cv::Mat imgL, cv::Mat imgR, int type, MaxNode* treeLeft,
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

int numDescendants(int indexL, int indexR, MaxNode* treeHorL, MaxNode* treeHorR) {

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

void clearPrevCurLevel(int width, int height, MaxNode* tree) {
	for (int i = 0; i < width * height; i++) {
		tree[i].prevCurLevel = false;
	}
}

void clearCurLevel(int width, int height, MaxNode* tree) {
	for (int i = 0; i < width * height; i++) {
		tree[i].prevCurLevel = tree[i].curLevel;
		tree[i].curLevel = false;
	}
}

void nthChildrenSide(int height, const std::vector<std::vector<int> >& topsL,
		int n, std::vector<std::vector<int> >* topsLnTh, MaxNode* treeLeftHor) {

	for (int r = 0; r < height; r++) {

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

		topsLnTh->push_back(rowLNew);
	}

}

void getNthTops(std::vector<std::vector<int>> topsR,
		std::vector<std::vector<int>> topsL, int n,
		std::vector<std::vector<int>> * topsRnTh,
		std::vector<std::vector<int>> * topsLnTh, MaxNode * treeRightHor,
		MaxNode * treeLeftHor, int height) {
	nthChildrenSide(height, topsL, n, topsLnTh, treeLeftHor);
	nthChildrenSide(height, topsR, n, topsRnTh, treeRightHor);
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
				matchesRowL[i].cost = std::numeric_limits<double>::infinity();
			}
			for (uint i = 0; i < matchesRowR.size(); i++) {
				matchesRowR[i].cost = std::numeric_limits<double>::infinity();
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

					calcCI(treeLeftHor, index, res, disparityLevels, width, height,kernelCostComp); // iteration==total-1?0:kernelCostComp

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
									if(treeLeftHor[index].Area > 4){
										dispMap->at<float>(r, c2+2) = min;
										dispMap->at<float>(r, c2_end-2) = min;
									}else{
										dispMap->at<float>(r, c2) = min;
										dispMap->at<float>(r, c2_end) = min;
									}

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



						std::vector<int> neighboursLTop = getNeighboursTop(
								treeLeftHor, indexL, width, height, true,
								false,kernelCostComp);
						std::vector<int> neighboursRTop = getNeighboursTop(
								treeRightHor, indexR, width, height, true,
								false,kernelCostComp);
						std::vector<int> neighboursLBottom =
								getNeighboursBottom(treeLeftHor, indexL, width,
										height, true, false,kernelCostComp);
						std::vector<int> neighboursRBottom =
								getNeighboursBottom(treeRightHor, indexR, width,
										height, true, false,kernelCostComp);

						float colCost = 0;
						float descendentsCost = 0;
						float sizeTop =
								neighboursLTop.size() > neighboursRTop.size() ?
										neighboursRTop.size() :
										neighboursLTop.size();
						float sizeBottom =
								neighboursLBottom.size()
										> neighboursRBottom.size() ?
										neighboursRBottom.size() :
										neighboursLBottom.size();


						for (int i = 0; i < sizeTop; i++) {

							float tmp = costComp(imgLeft, imgRight, 0,
									treeLeftHor, treeRightHor,
									neighboursLTop[i], neighboursRTop[i]);
							colCost += tmp;

							tmp = numDescendants(neighboursLTop[i],neighboursRTop[i],treeLeftHor,treeRightHor);
							descendentsCost += tmp;
						}

						for (int i = 0; i < sizeBottom; i++) {
							float tmp = costComp(imgLeft, imgRight, 0,
									treeLeftHor, treeRightHor,
									neighboursLBottom[i], neighboursRBottom[i]);
							colCost += tmp;

							tmp = numDescendants(neighboursLBottom[i],neighboursRBottom[i],treeLeftHor,treeRightHor);
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

						neighboursLTop.clear();
						neighboursRTop.clear();
						neighboursLBottom.clear();
						neighboursRBottom.clear();

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
						== std::numeric_limits<double>::infinity()) {
					continue;
				}

				Match L = matchesRowL[LvInd];

				if (matchesRowR[L.match].cost
						== std::numeric_limits<double>::infinity())
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

void coarseRows(MaxNode *treeRightHor, MaxNode *treeLeftHor,
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





class BuildTreeParalel: public cv::ParallelLoopBody {

private:
	int height;
	int width;
	cv::Mat gval;
	MaxNode* node;
public:

	BuildTreeParalel(int height, int width,cv::Mat gval,MaxNode * node) : height(height), width(width), gval(gval),node(node) {
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



MaxNode* buildTree(int height, int width, cv::Mat gval){

	MaxNode* node = (MaxNode*)malloc(width*height*sizeof(MaxNode));

	parallel_for_(cv::Range(0, height), BuildTreeParalel(height, width, gval,node));

	return node;
}

/****** Main function ******/

int main (int argc, char *argv[]){

	const char *filenameLeftImage = argv[1];
	const char *filenameRightImage = argv[2];
	int maxDisplacement = atoi(argv[3]);
	const char *outFolder = argv[4];
	filenameOutput = argv[5];

	double startTotal = getWallTime();

	outFolderGV = outFolder;
	disparityLevels = maxDisplacement;
	cv::Mat imgLeftResized;
	cv::Mat imgRightResized;
	int width, height;
	cv::Mat ones;

	// read left & right images

	cv::Mat imgLeft = cv::imread(filenameLeftImage, -1);
	cv::Mat imgRight = cv::imread(filenameRightImage, -1);

	maxAreaMatchedFineTopNode = imgLeft.cols/factorMax;

	// resize left & right images

	disparityLevelsResized = disparityLevels;

	// convert left & right images to grayscale

	cv::cvtColor(imgLeft, imgLeft, cv::COLOR_RGB2GRAY);
	cv::cvtColor(imgRight, imgRight, cv::COLOR_RGB2GRAY);

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

	width = imgLeft.cols;
	height = imgLeft.rows;

	treeLeftHor = buildTree(height,width,sobelLeft3ucharHor);
	treeRightHor = buildTree(height,width,sobelRight3ucharHor);

	sobelLeft3ucharHor.release();
	sobelRight3ucharHor.release();
	sobelLeft3ucharVert.release();
	sobelRight3ucharVert.release();

	// Get tops

	std::vector<std::vector<int>> topsL;
	std::vector<std::vector<int>> topsR;

	std::vector<std::vector<int>> topsLnTh;
	std::vector<std::vector<int>> topsRnTh;
	std::vector<std::vector<int>> topsLnThPrev;
	std::vector<std::vector<int>> topsRnThPrev;

	getTops(treeRightHor, treeLeftHor, width, height, &topsL, &topsR);

	// initialize dispmap with -1

	cv::Mat dispMap;
	dispMap = cv::Mat::ones(cv::Size(imgLeft.cols, imgLeft.rows), CV_32F);
	dispMap *= -1;

	// coarse to fine levels

	for (int i = 0; i < total; i++) {

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

		getNthTops(topsR, topsL, sizes[i], &topsRnTh, &topsLnTh, treeRightHor,
				treeLeftHor, height);

		 coarseRows(treeRightHor, treeLeftHor, disparityLevels, imgRight2, imgLeft2, topsLnTh, topsRnTh, i, &dispMap);

		if (i > 0) {
			clear2Dvect(topsLnThPrev);
			clear2Dvect(topsRnThPrev);
		}

		topsLnThPrev = topsLnTh;
		topsRnThPrev = topsRnTh;

		topsLnTh = (std::vector<std::vector<int>>) 0;
		topsRnTh = (std::vector<std::vector<int>>) 0;
	}

	if (sparse) {
		int kermel =21;
		cv::Mat dispMapFinal = dispMap.clone();

		for(int r = 0;r<dispMap.rows-0;r++){
			for(int c = 0;c<dispMap.cols-0;c++){
					float curval = dispMap.at<float>(r,c);
					if(curval != -1){
						int total = 0;
						int totalMean = 0;
						for(int rOff = r-kermel>0?-kermel:0;rOff<(r+kermel<=dispMap.rows?kermel:0);rOff++){
							for(int cOff = c-kermel>0?-kermel:0;cOff<(c+kermel<=dispMap.cols?kermel:0);cOff++){
								if(dispMap.at<float>(r+rOff,c+cOff) != -1){
									if(abs(dispMap.at<float>(r+rOff,c+cOff) - curval) <= abs(cOff)){
										total++;
									}else{
										totalMean++;
									}
								}
							}
						}
						if(totalMean != 0 && total < totalMean)
							dispMapFinal.at<float>(r,c) = -1;
					}
			}
		}
		dispMap = dispMapFinal;
	}

	plotDispmap(dispMap, disparityLevelsResized, true);


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
	dispMap.release();
	free(treeLeftHor);
	free(treeRightHor);

	double endTotal = getWallTime();
	double totalTime = (endTotal - startTotal);
	double MPS = ((float) width / (float) 1000.0)
			* ((float) height / (float) 1000.0);

	printf("runtime: %.4fs (%.4f/MP)\n", totalTime, totalTime / MPS);


	return 0;
}
