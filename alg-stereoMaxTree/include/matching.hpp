/**
 * Max Tree Stereo Matching
 * matching.h
 *
 * Matching module of the program, it holds all function useful to evaluate
 * a cost function and execute the matching algorithm on two stereo images.
 *
 * @author: A.Fortino
 */

#ifndef INCLUDE_MATCHING_HPP_
#define INCLUDE_MATCHING_HPP_

#include "config.hpp"
#include "manageThreads.hpp"

void run(int numThreads, const char* filenameLeftImage,
		const char* filenameRightImage, int disparityLevels,
		const char* filenameImageOut, int maxDisplacement, int ksize,
		int convert, int typeVolumeFiltering, int typeDisparityFiltering,
		int dispKernelSize, const char* filenameDisp, const char* testcase, const char * filename) ;
cv::Mat computeCostVolume(MaxNode* treeLeft, MaxNode* treeRight, MaxNode* treeMinLeft, MaxNode* treeMinRight, LevelRoot** leftRoots,
									int* sizeRoots, int width, int height, int disparityLevels);
void saveDisparityImage(const char* filename, const char* filedispname, int* disparityMap, int disparityLevels,
							int width, int height);
int* computeDisparity(cv::Mat costVolume, LevelRoot** leftRoots, int* sizeRoots, MaxNode* treeLeft,
					int width, int height, int disparityLevels, int maxDisplacement);
float costFunctionArea(MaxNode* treeLeft, MaxNode* treeRight, MaxNode* treeMinLeft, MaxNode* treeMinRight, int idLeftNode, int idRightNode, int width);
float costFunctionColor(MaxNode* treeLeft, MaxNode* treeRight, MaxNode* treeMinLeft, MaxNode* treeMinRight, int idLeftNode, int idRightNode);
void filterCostVolume(cv::Mat costVolume, int ksize, int typeFiltering);
int compareInt (const void * a, const void * b);

#endif /* INCLUDE_MATCHING_HPP_ */
