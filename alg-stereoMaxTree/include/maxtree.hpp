/**
 * Max Tree Stereo Matching
 * maxtree.c
 *
 * Max Tree module of the program. This module implements all functions needed to
 * build max tree and identify level roots inside the trees.
 *
 * @author: A.Fortino
 */

#ifndef INCLUDE_MAXTREE_HPP_
#define INCLUDE_MAXTREE_HPP_

#include "config.hpp"

typedef struct LevelRoot {
	int nodeID;
	int graylevel;
} LevelRoot;

void BuildMaxTree1D(int row, Pixel *gval, int width, int height, MaxNode *node);
void BuildMinTree1D(int row, Pixel *gval, int width, int height, MaxNode *node);
LevelRoot* FindLevelRoots (int *size, MaxNode *tree, int width, int row);
LevelRoot* InsertRootInOrder(LevelRoot* roots, int *size, int graylevel, int nodeID);
void sortLevelRoots(LevelRoot* roots, int low, int high);
int partition(LevelRoot* roots, int low, int high);

#endif /* INCLUDE_MAXTREE_HPP_ */
