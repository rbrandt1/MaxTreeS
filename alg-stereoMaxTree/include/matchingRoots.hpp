/*
 * matchingRoots.h
 *
 *  Created on: 14 giu 2018
 *      Author: amedeo
 */

#ifndef INCLUDE_MATCHINGROOTS_HPP_
#define INCLUDE_MATCHINGROOTS_HPP_

#include "config.hpp"
#include "manageThreads.hpp"

MaxNode* computeCostMapNormalized(MaxNode* treeLeft, MaxNode* treeRight, int width, int height, int maxShift);
int costFunctionRoots(MaxNode* treeLeft, MaxNode* treeRight, LevelRoot* leftRoots,
								LevelRoot* rightRoots, int leftIndex, int rightIndex,
								int i, int width, int numleftRoots, int numrightRoots);

int costFunctionColorRoots(MaxNode* treeLeft, MaxNode* treeRight, LevelRoot* leftRoots,
		LevelRoot* rightRoots, int leftIndex, int rightIndex,
		int i, int width, int numleftRoots, int numrightRoots);
int costFunctionAreaRoots(MaxNode* treeLeft, MaxNode* treeRight, LevelRoot* leftRoots,
								LevelRoot* rightRoots, int leftIndex, int rightIndex,
								int i, int width, int numleftRoots, int numrightRoots);
int computeAreaDiff(MaxNode* treeLeft, MaxNode* treeRight, int idLeftNode, int idRightNode, int width);
int computeColorDiff(MaxNode* treeLeft, MaxNode* treeRight, int idLeftNode, int idRightNode);


#endif /* INCLUDE_MATCHINGROOTS_HPP_ */
