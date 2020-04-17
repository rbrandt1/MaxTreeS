#include "core.hpp"

/****** Parameters ******/

float alpha = .8; // alpha
int minAreaMatchedFineTopNode = 0; // omega_alpha
int maxAreaMatchedFineTopNode; // omega_alpha
float factorMax = 3; // omega_beta = image_width/factorMax
int nColors = 5; // q
int sizes[16] = {1,0,0,0}; // S, appended with 0,0 which correspond to the refinement step
int total = 4; // #S + 2
uint kernelCostComp = 6; // omega_delta
bool sparse = true; // produce sparse (true) or semi-dense (false) disparity map
int everyNthRow = 1;

const char * outFolderGV;

bool skipsides = true;
const char* filenameOutput;
int disparityLevels;

float minConfidencePercentage;
float allowanceSearchRange;

cv::Mat costVolume[1000];

bool printTimesBool = true;

bool saveStageImages = false;

bool useLRCHECK = true;
float uniquenessRatio = 12;

bool print1 = false;
bool print2 = false;
bool print3 = false;
bool print4 = false;
bool print5 = false;
bool print6 = false;
bool print7 = false;
bool print8 = false;
bool print9 = false;


int timeCostVolAcess_start = 0;
int timeCostVolAcess_end = 0;

double timeCostVolAcess_totaltime = 0;
double timeCostVolAcess_totaltime_2 = 0;

/****** Functions ******/

class ParallelMatching : public cv::ParallelLoopBody{
public:
  ParallelMatching (cv::Mat & dispMap, cv::Mat & dispMapBackup, const int width, const int maxDisp,const float uniquenessRatio)
  : dispMap(dispMap),dispMapBackup(dispMapBackup), width(width), maxDisp(maxDisp),uniquenessRatio(uniquenessRatio){}
  virtual void operator()(const cv::Range & range) const {
    for (int i = range.start; i < range.end; i++){

        int r = i / dispMap.cols;
        int c = i % dispMap.cols;

        if(r % everyNthRow != 0)
          continue;

        if(c > maxDisp){
          int indexMin_Left = 0;
          float bestCost = 10000000;
          float secondBestCost = 10000000;

          float factor = 0.15;
          float dispVal = dispMapBackup.ptr<short>(r)[c];

          if(dispVal > 1){
          	 int newMinDisp = dispVal * (1-factor)  > 1 ? dispVal * (1-factor) : 1;
         	 int newMaxDisp = dispVal * (1+factor) < maxDisp ? dispVal * (1+factor) : maxDisp;

         	 for(int d = newMinDisp; d < newMaxDisp; d++){
	   
			float *costData = (float*)(costVolume[d].data);
			float cost = costData[costVolume[d].cols * (r) + (c-maxDisp)];

	            if(cost <= bestCost){
	              secondBestCost = bestCost;
	              bestCost = cost;
	              indexMin_Left = d;      
	            }
	          }
	          if(uniquenessRatio/100 > (secondBestCost-bestCost)/bestCost){
	            indexMin_Left = 0;
	          }
	          dispMap.ptr<short>(r)[c] = indexMin_Left;
          }
          if(print1){print1=false;printf("Pixelmatching: uniquenessRatio: %f factor: %f\n",uniquenessRatio,factor);}
        }
		
		
		
		
      }
  }
  ParallelMatching& operator=(const ParallelMatching &) {
    return *this;
  };
private:
  cv::Mat & dispMap;
  cv::Mat & dispMapBackup;
  int width;
  int maxDisp;
 float uniquenessRatio;
};




class BuildCostVolumeParallel : public cv::ParallelLoopBody{
public:
  BuildCostVolumeParallel (const int kernelSize, const cv::Mat imlL, const cv::Mat imr, const int maxDisp )  : kernelSize(kernelSize),imlL(imlL),imr(imr), maxDisp(maxDisp){}
  virtual void operator()(const cv::Range & range) const {
    for (int d = range.start; d < range.end; d++){
	costVolume[d] =	abs(imlL-imr(cv::Rect(maxDisp-d,0,imr.cols-maxDisp,imr.rows)));
	cv::cvtColor(costVolume[d], costVolume[d], cv::COLOR_RGB2GRAY);
	costVolume[d].convertTo(costVolume[d],CV_32F);
        cv::GaussianBlur(costVolume[d],costVolume[d],cv::Size(kernelSize,kernelSize),0);
		
		
		if(print9){print9=false;
		
			printf("BuildCostVolumeParallel: kernelSize: %d \n",kernelSize);
		
		}
		
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







void buildCostVolume(cv::Mat iml,cv::Mat imr, int maxDisp,int kernelSize){
	
	cv::Rect roiL = cv::Rect(maxDisp,0,iml.cols-maxDisp,imr.rows);
	cv::Mat imlL = iml(roiL);

	BuildCostVolumeParallel buildCostVolumeParallel(kernelSize, imlL, imr, maxDisp);
	cv::parallel_for_(cv::Range(0, maxDisp), buildCostVolumeParallel);
}





void writeVisDispmap(cv::Mat img, std::string str, float dispLevels) {

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

		index = index - width;

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


	if(print6){print6=false;
		
		printf("getNeighborsTop: kernelCostComp: %d \n",kernelCostComp);
		
	}

	std::vector<int> tmp;

	int prevIndex = indexCenter;

	if (!dispCheck
			|| (tree[prevIndex].dispLprev >= 0 && tree[prevIndex].dispRprev >= 0))
		tmp.push_back(prevIndex);

	for (int row = 0; tmp.empty() || tmp.size() < kernelCostComp; row++) {
		int index = tree[prevIndex].begIndex + tree[prevIndex].Area / 2;

		index = index + width;

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
		neighTop = getNeighborsTop(tree, index, width, height, false, true, kernel);
		neighBottom = getNeighborsBottom(tree, index, width, height, false, true, kernel);
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


float gradientCostComp(int y2, int xL1, int xR1, int xL2, int xR2,int maxDisp) {
	


	int width = costVolume[1].cols;
	int height = costVolume[1].rows;
	
	if(y2< 0 || y2 >= height || xR1-xL1 <= 0 || xR1-xL1 >= width)
		return 100000;
	
	float total = 0;
	float items = 0;
	for(float ratio = 0;ratio <= 1.0;ratio += 1.0/((float)(xR1-xL1))){
		int disp = abs(xL1-xL2)*ratio + abs(xR1-xR2)*(1-ratio);

		

		if(disparityLevels > disp && disp >= 0) {
			int xIndex = xL1 * ratio + (1-ratio) * xR1;
			if(0 <= xIndex-maxDisp && xIndex-maxDisp < width){
				float *costData = (float*)(costVolume[disp].data);
				total += costData[width * (y2) + (xIndex-maxDisp)];
				items++;
			}
		}
	}
	if(items > 0){
		return total / items;
	}else{
		return 100000;
	}
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

			sum += weight * fabs(((float)treeHorL[indexL].Area / (float)(treeHorL[indexL].Area+treeHorR[indexR].Area ))-.5 );

			indexL = treeHorL[indexL].parent;
			indexR = treeHorR[indexR].parent;

			total += weight;
			//weight/=2;
		}
		
		if(print4){print4=false;
		
			printf("contextCost: returnval: %f\n",(225)*(sum / total));
		
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


		if(print5){print5=false;
		
			printf("getTops: minAreaMatchedFineTopNode: %d maxAreaMatchedFineTopNode:%d\n",minAreaMatchedFineTopNode,maxAreaMatchedFineTopNode);
		
		}


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

	NthChildrenSide nthChildrenSide(topsL, n, topsLnTh, treeLeftHor);
	cv::parallel_for_(cv::Range(0, height), nthChildrenSide);

	NthChildrenSide nthChildrenSide2(topsR, n, topsRnTh, treeRightHor);
	cv::parallel_for_(cv::Range(0, height), nthChildrenSide2);

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

			
			timeCostVolAcess_start++;
			auto timeCostVolAcess_start_time = getWallTime();

			int * res = (int *) malloc(2 * sizeof(int));
			if(iteration < total -2 && r % everyNthRow != 0)
				continue; 
			

			std::vector<Match> matchesRowL(topsL[r].size());
			std::vector<Match> matchesRowR(topsR[r].size());


			if (total - 2 > iteration){
				for (uint i = 0; i < matchesRowL.size(); i++) {
					matchesRowL[i].cost = std::numeric_limits<double>::infinity();
				}
				for (uint i = 0; i < matchesRowR.size(); i++) {
					matchesRowR[i].cost = std::numeric_limits<double>::infinity();
				}
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
					if(total - 2 <= iteration){
						if ( total - 2 == iteration) {
							float lDisp = treeLeftHor[index].dispLprev;
							float rDisp = treeLeftHor[index].dispRprev;
							
							for (int column = c2; column <= c2_end; column++) {
								float ratio = 1;
								if(c2_end-c2 != 0)
									ratio = (c2_end-column)/(c2_end-c2);
							
								short *dispMapData = (short*)(dispMap->data);
								dispMapData[width * (r) + (column)] = ratio*rDisp+(1-ratio)*lDisp;
							}
						}
					}else{
						

						calcDispSearchRange(treeLeftHor, index, res, disparityLevels, width, height,kernelCostComp);

						int n0 = res[0];
						int n1 = res[1];

						begBeg = c2_parent-n0 >=0?c2_parent-n0:0;
						endEnd = c2_end_parent-n1 >=0?c2_end_parent-n1:0;

						
					}				
				} else {
					begBeg = fromBeg;
					endEnd = toEnd;
				}


				if (total - 2 <= iteration){
						continue;
				}


				for (uint RvInd = 0;  RvInd < topsR[r].size(); RvInd++) {

					int c = treeRightHor[topsR[r][RvInd]].begIndex % width;
					int c_end = treeRightHor[topsR[r][RvInd]].begIndex % width
							+ treeRightHor[topsR[r][RvInd]].Area - 1;

					if (c2 > disparityLevels && c <= c2 && c_end <= c2_end
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
						
							int y = height - 1 - treeLeftHor[NeighborsLTop[i]].begIndex / width;
							int xL1 = (treeLeftHor[NeighborsLTop[i]].begIndex )% width;
							int xR1 = (treeLeftHor[NeighborsLTop[i]].Area % width + treeLeftHor[NeighborsLTop[i]].begIndex % width) % width;
							int xL2 = (treeRightHor[NeighborsRTop[i]].begIndex) % width;
							int xR2 = (treeRightHor[NeighborsRTop[i]].Area % width + treeRightHor[NeighborsRTop[i]].begIndex % width) % width;
							

							auto timeCostVolAcess_start_time_2 = getWallTime();


							float grad2 = gradientCostComp(y,xL1,xR1,xL2,xR2,disparityLevels);
							colCost += grad2;

							timeCostVolAcess_totaltime_2 += getWallTime() - timeCostVolAcess_start_time_2;
													
							float tmp = contextCost(NeighborsLTop[i],NeighborsRTop[i],treeLeftHor,treeRightHor);
							descendentsCost += tmp;
						}
						
			

						for (int i = 0; i < sizeBottom; i++) {
							int y = height - 1 - treeLeftHor[NeighborsLBottom[i]].begIndex / width;
							int xL1 = (treeLeftHor[NeighborsLBottom[i]].begIndex) % width;
							int xR1 = ( treeLeftHor[NeighborsLBottom[i]].Area % width + treeLeftHor[NeighborsLBottom[i]].begIndex % width) % width;
							int xL2 = (treeRightHor[NeighborsRBottom[i]].begIndex) % width;
							int xR2 = ( treeRightHor[NeighborsRBottom[i]].Area % width + treeRightHor[NeighborsRBottom[i]].begIndex % width) % width;
							

							auto timeCostVolAcess_start_time_2 = getWallTime();


							float grad2 = gradientCostComp(y,xL1,xR1,xL2,xR2,disparityLevels);
							colCost += grad2;
				

							timeCostVolAcess_totaltime_2 += getWallTime() - timeCostVolAcess_start_time_2;


							float tmp = contextCost(NeighborsLBottom[i],NeighborsRBottom[i],treeLeftHor,treeRightHor);
							descendentsCost += tmp;
						}

						cost += (colCost / (sizeBottom + sizeTop - 1))
								* alpha;
						cost += (descendentsCost / (sizeBottom + sizeTop - 1))
								* (1 - alpha);
								
						if(print3){print3=false;
						
							printf("MatchTopsParallelbody: alpha: %f\n",alpha);
							
							//printf("MatchTopsParallelbody: colcost: %f\n",(colCost / (sizeBottom + sizeTop - 1))* alpha);
							//printf("MatchTopsParallelbody: descendentsCost: %f\n",(descendentsCost / (sizeBottom + sizeTop - 1))* (1 - alpha));
						
						}

						if (matchesRowL[LvInd].cost > cost) {
							matchesRowL[LvInd].secondBestCost = matchesRowL[LvInd].cost;
							matchesRowL[LvInd].cost = cost;
							matchesRowL[LvInd].match = RvInd;
						};
						if (matchesRowR[RvInd].cost > cost) {
							matchesRowR[RvInd].secondBestCost = matchesRowR[RvInd].cost;
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

			if (total - 2 <= iteration){

				timeCostVolAcess_end++;
				timeCostVolAcess_totaltime += getWallTime() - timeCostVolAcess_start_time;

				free(res);
				continue;			
			}
					

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


				if (!(R.secondBestCost < R.cost * (1+uniquenessRatio/100)) && !(L.secondBestCost < L.cost * (1+uniquenessRatio/100)) && (R.match == tmR || !useLRCHECK)) {

					if(print2){print2=false;printf("MatchTopsParallelbody: uniquenessRatio: %f\n",uniquenessRatio);}

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
			free(res);	


			timeCostVolAcess_end++;
			timeCostVolAcess_totaltime += getWallTime() - timeCostVolAcess_start_time;
	
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

	parallel_for_(cv::Range(0, height), BuildTreeParalel(height, width, gval,node));

	return node;
}


class NoiseFilter : public cv::ParallelLoopBody{
public:
  NoiseFilter (short * dispMap,short * dispMapFinal,int cols,int rows, const int kermel)
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
				short absCoff = abs(cOff-c);
				for (int rOff = rBeg; rOff < rEnd; rOff++) {
					short val = dispMap[cols * (rOff) + (cOff)];
					if (val > 1) {
					    if (abs(val - curval) <= absCoff) {
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
	const int kermel;
};

auto time0 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
std::string prevString = ".";
void printTimes(std::string stringCur){
	if(printTimesBool){
		auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
		printf("%0.f ---- %s\n",(double)(time1-time0),prevString.c_str());
		prevString = stringCur;
		time0 = time1;
	}
}


void saveStageImage(cv::Mat tmp2, std::string filename){
	if(saveStageImages){
		cv::Mat tmp = tmp2.clone();
		tmp.convertTo(tmp, CV_8U);
		cv::applyColorMap(tmp, tmp, cv::COLORMAP_JET);		
		cv::imwrite(filename.c_str(),tmp);
	}
}



class InterpolateParallel : public cv::ParallelLoopBody{
public:
  InterpolateParallel ( MaxNode *treeLeftHor ,int disparityLevels,int kernelCostComp,std::vector<std::vector<int>> topsL,cv::Mat & dispMap)  : treeLeftHor(treeLeftHor) ,disparityLevels(disparityLevels),kernelCostComp(kernelCostComp),topsL(topsL),dispMap(dispMap) {}
  virtual void operator()(const cv::Range & range) const {
    for (int r = range.start; r < range.end; r++){
	  int * res = (int *) malloc(2 * sizeof(int));
	  for (int LvInd = 0; LvInd < topsL[r].size(); LvInd++) {
		 int index = topsL[r][LvInd];
		
		 
		 calcDispSearchRange(treeLeftHor, index, res, disparityLevels, dispMap.cols, dispMap.rows, kernelCostComp);

		 int c2 = treeLeftHor[index].begIndex % dispMap.cols;
		 int c2_end = treeLeftHor[index].begIndex % dispMap.cols + treeLeftHor[index].Area - 1;

		float lDisp = res[0];
		float rDisp = res[1];

		if(lDisp > 0 && lDisp < disparityLevels && rDisp > 0 && rDisp < disparityLevels){
			if(sparse){
				int offset = 0;
				if(treeLeftHor[topsL[r][LvInd]].Area > 4)
					offset = 2;
				dispMap.ptr<short>(r)[c2+offset] = lDisp;
				dispMap.ptr<short>(r)[c2_end-offset] = rDisp;
			}else{
				for (int column = c2; column <= c2_end; column++) {
					float ratio = 1;
					if(c2_end-c2 != 0)
						ratio = (c2_end-column)/(c2_end-c2);
					dispMap.ptr<short>(r)[column] = ratio*rDisp+(1-ratio)*lDisp;
				}
			}
		}		
	  }
     }
  }
  InterpolateParallel& operator=(const InterpolateParallel &) {
    return *this;
  };
private:
	MaxNode *treeLeftHor;
	int disparityLevels;
	int kernelCostComp;
	std::vector<std::vector<int>> topsL;
	cv::Mat & dispMap;
};



cv::Mat work(cv::Mat imgLeft, cv::Mat imgRight,bool sparse2,float alpha2,int minAreaMatchedFineTopNode2,int maxAreaMatchedFineTopNode2,int nColors2,int * sizes2, int total2,uint kernelCostComp2,int kernelCostVolume, int everyNthRow2, int disparityLevels2,float minConfidencePercentage2, float allowanceSearchRange2){
	
/*

	print1 = true;
	print2 = true;
	print3 = true;
	print4 = true;
	print5 = true;
	print6 = true;
	print7 = true;
	print8 = true;
	print9 = true;


	print1 = false;
	print2 = false;
	print3 = false;
	print4 = false;
	print5 = false;
	print6 = false;
	print7 = false;
	print8 = false;
	print9 = false;
*/


	timeCostVolAcess_start = 0;
	timeCostVolAcess_end = 0;
	timeCostVolAcess_totaltime = 0;
	timeCostVolAcess_totaltime_2 = 0;

	printTimes("copy variables");
	
	sparse= sparse2;
	alpha= alpha2;
	minAreaMatchedFineTopNode= minAreaMatchedFineTopNode2;
	maxAreaMatchedFineTopNode= maxAreaMatchedFineTopNode2;
	nColors= nColors2;
	for(int i =0;i<total2;i++){
		sizes[i] = sizes2[i];
	}
	total = total2;
	kernelCostComp = kernelCostComp2;
	everyNthRow= everyNthRow2;
	minConfidencePercentage = minConfidencePercentage2;
	allowanceSearchRange = allowanceSearchRange2;
	disparityLevels = disparityLevels2;
	int width, height;
	width = imgLeft.cols;
	height = imgLeft.rows;
	cv::Mat ones;

	double startTotal = getWallTime();

	printTimes("Apply median filter to left & right image");
	
	cv::medianBlur(imgLeft,imgLeft,5);
	cv::medianBlur(imgRight,imgRight,5);



	printTimes("Initialize disparity map with -1");
	
	cv::Mat dispMap;
	dispMap = cv::Mat::ones(cv::Size(imgLeft.cols, imgLeft.rows), CV_16S);
	dispMap *= -1;


	printTimes("Compute trees");

	MaxNode *treeLeftHor = NULL;
	MaxNode *treeRightHor = NULL;
	cv::Mat sobelLeft3Hor;
	cv::Mat sobelRight3Hor;
	cv::Mat leftG;
	cv::Mat rightG;
	cv::Mat sobelLeft3Vert;
	cv::Mat sobelRight3Vert;
	cv::Mat sobelLeftHor;
	cv::Mat sobelRightHor;
	cv::Mat sobelLeftVert;
	cv::Mat sobelRightVert;

	cv::Sobel(imgLeft, sobelLeftHor, 5, 1, 0);
	cv::Sobel(imgRight, sobelRightHor, 5, 1, 0);

	cv::Sobel(imgLeft, sobelLeftVert, 5, 0, 1);
	cv::Sobel(imgRight, sobelRightVert, 5, 0, 1);

	sobelLeft3Hor = (abs(sobelLeftHor) + abs(sobelLeftVert)) / 2;
	sobelRight3Hor = (abs(sobelRightHor) + abs(sobelRightVert)) / 2;

	ones = cv::Mat::ones(cv::Size(imgLeft.cols, imgLeft.rows), CV_32F);

	sobelLeft3Hor = (255.0 * ones - sobelLeft3Hor);
	sobelRight3Hor = (255.0 * ones - sobelRight3Hor);

	float fmin = 125;
	sobelLeft3Hor = ((sobelLeft3Hor - fmin)/(255-fmin))*255;
	sobelRight3Hor  = ((sobelRight3Hor - fmin)/(255-fmin))*255;

	sobelLeft3Hor.convertTo(leftG, CV_8U);
	sobelRight3Hor.convertTo(rightG, CV_8U);

	sobelLeft3Hor.release();
	sobelRight3Hor.release();

	leftG = leftG / (256 / nColors);
	rightG = rightG / (256 / nColors);
	leftG = leftG * (256 / nColors);
	rightG = rightG * (256 / nColors);

	
	if(print7){print7=false;
	
		printf("work: nColors: %d\n",nColors);
	
	}


	treeLeftHor = buildTree(leftG);
	treeRightHor = buildTree(rightG);

	if(saveStageImages){		
		cv::imwrite("stage1.png",leftG);
	}

	leftG.release();
	rightG.release();

	printTimes("Compute cost volume");

	cv::Mat imgLeft2;
	cv::Mat imgRight2;
	std::vector<cv::Mat> channelsL;
	std::vector<cv::Mat> channelsR;

	cv::Mat imL2;
	imgLeft.convertTo(imL2,CV_32F);
	cv::Mat imR2;
	imgRight.convertTo(imR2,CV_32F);

	channelsL.push_back(imL2);
	channelsR.push_back(imR2);


	cv::Sobel(imgLeft, sobelLeftHor, 3, 1, 0);
	cv::Sobel(imgRight, sobelRightHor, 3, 1, 0);

	cv::Mat slh;
	sobelLeftHor.convertTo(slh,CV_32F);
	cv::Mat srh;
	sobelRightHor.convertTo(srh,CV_32F);

	channelsL.push_back(slh);
	channelsR.push_back(srh);

	cv::Sobel(imgLeft, sobelLeftVert, 3, 0, 1);
	cv::Sobel(imgRight, sobelRightVert, 3, 0, 1);

	cv::Mat slv;
	sobelLeftVert.convertTo(slv,CV_32F);
	cv::Mat srv;
	sobelRightVert.convertTo(srv,CV_32F);

	channelsL.push_back(slv);
	channelsR.push_back(srv);

	merge(channelsL, imgLeft2);
	merge(channelsR, imgRight2);



	buildCostVolume(imgLeft2,imgRight2, disparityLevels, kernelCostVolume);



	printTimes("Release allocated data");

	ones.release();
	sobelLeft3Hor.release();
	sobelRight3Hor.release();
	sobelLeft3Vert.release();
	sobelRight3Vert.release();
	sobelLeftHor.release();
	sobelRightHor.release();
	sobelLeftVert.release();
	sobelRightVert.release();

	printTimes("getTops");

	std::vector<std::vector<int>> topsL;
	std::vector<std::vector<int>> topsR;

	std::vector<std::vector<int>> topsLnTh(height);
	std::vector<std::vector<int>> topsRnTh(height);
	std::vector<std::vector<int>> topsLnThPrev(height);
	std::vector<std::vector<int>> topsRnThPrev(height);

	getTops(treeRightHor, treeLeftHor, width, height, &topsL, &topsR);


	
	for (int i = 0; i < total-1; i++) {

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

		printTimes("getNthTops");

		getNthTops(topsR, topsL, sizes[i], &topsRnTh, &topsLnTh, treeRightHor, treeLeftHor, height);

		printTimes("coarse to fine");

		matchNodes(treeRightHor, treeLeftHor, disparityLevels, imgRight2, imgLeft2, topsLnTh, topsRnTh, i, &dispMap);
		
		if(i < total - 2){
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

	saveStageImage(dispMap, "stage2.png");

	printTimes("Noise filter");

	cv::Mat dispMapFinal = dispMap.clone();

	short *dispMapData = (short*)(dispMap.data);
	short *dispMapFinalData = (short*)(dispMapFinal.data);

	NoiseFilter noiseFilter(dispMapData,dispMapFinalData,dispMap.cols,dispMap.rows, 21);
	cv::parallel_for_(cv::Range(0, dispMap.rows), noiseFilter);
	dispMap.release();
	dispMap = dispMapFinal;

	saveStageImage(dispMap, "stage3.png");


	printTimes("Assign to tree");
	
	for (int i = 0; i < dispMap.rows*dispMap.cols; i++){
		int r = i / dispMap.cols;
		int c = i % dispMap.cols;

		int index = (dispMap.rows - 1 - r) * dispMap.cols + c;
		if(  ! treeLeftHor[index].levelroot ){
			index = treeLeftHor[index].parent;      
		}
		if(index == ROOT)
			continue;

		int c2 = treeLeftHor[index].begIndex % dispMap.cols;
		int c2_end = treeLeftHor[index].begIndex % dispMap.cols + treeLeftHor[index].Area - 1;

		treeLeftHor[index].dispLprev = dispMap.ptr<short>(r)[c2];
		treeLeftHor[index].dispRprev = dispMap.ptr<short>(r)[c2_end];
	}

	dispMap = dispMap * 0;

	printTimes("Interpolate, assign to dispmap");

	InterpolateParallel interpolateParallel(treeLeftHor, disparityLevels, kernelCostComp, topsL,dispMap);
	cv::parallel_for_(cv::Range(0, dispMap.rows), interpolateParallel);


	saveStageImage(dispMap, "stage4.png");

	printTimes("Free treess");

	free(treeLeftHor);
	free(treeRightHor);

	printTimes("Guided pixel matching");
		
	cv::Mat dispMap2 = cv::Mat::ones(height,width, CV_16S);
	dispMap2 = dispMap2 * -1;

	ParallelMatching parallelMatching(dispMap2,dispMap, width, disparityLevels, sparse ? uniquenessRatio : 4);
	cv::parallel_for_(cv::Range(0, dispMap2.rows*dispMap2.cols), parallelMatching);
		
	dispMap.release();
	dispMap = dispMap2; 
	

	saveStageImage(dispMap, "stage5.png");
	
	printTimes("Noise filter");
	
	dispMapFinal.release();
	dispMapFinal = dispMap.clone();

	dispMapData = (short*)(dispMap.data);
	dispMapFinalData = (short*)(dispMapFinal.data);

	NoiseFilter noiseFilter2(dispMapData,dispMapFinalData,dispMap.cols,dispMap.rows, 21);
	cv::parallel_for_(cv::Range(0, dispMap.rows), noiseFilter2);
	dispMap.release();
	dispMap = dispMapFinal;

	printTimes(".");

	//printf("time: %f\n",getWallTime() - startTotal);

	saveStageImage(dispMap, "stage6.png");

	dispMap.convertTo(dispMap,CV_32F);


	printf("timeCostVolAcess_start: %d\n",timeCostVolAcess_start);
	printf("timeCostVolAcess_end: %d\n",timeCostVolAcess_end);
	printf("timeCostVolAcess_totaltime: %.0f\n",timeCostVolAcess_totaltime*1000);
	printf("timeCostVolAcess_totaltime_2: %.0f\n",timeCostVolAcess_totaltime_2*1000);
	printf("Percentage: %.0f\n",(timeCostVolAcess_totaltime_2/timeCostVolAcess_totaltime)*100);



	return dispMap;
}
