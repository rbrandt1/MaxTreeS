/*
 * matchingRoots.c
 *
 *  Created on: 14 giu 2018
 *      Author: amedeo
 */
#include "include/matchingRoots.hpp"

MaxNode* computeCostMapNormalized(MaxNode* treeLeft, MaxNode* treeRight, int width, int height, int maxShift){
	int numRootsLeft, numRootsRight;
		LevelRoot* leftRoots, *rightRoots;
		MaxNode* treeout;
		double currentAreaCost, currentColorCost;
		double tmpColorCost, tmpAreaCost;
		double *costColor, *costArea;
		double maxAreaCost, maxColorCost;
		int rightIndex, idLeftNode, numCandidates, parentRootId;

		/**
		 * Copy of the tree structure
		 */
		treeout = (MaxNode*)calloc(width*height, sizeof(MaxNode));
		memcpy(treeout, treeLeft, (width*height)*sizeof(MaxNode));

		/**
		 * Cost matrices
		 */
		costColor = (double*)calloc(width*height, sizeof(double));
		costArea = (double*)calloc(width*height, sizeof(double));

		/**
		 * For each scanline, find level roots and compute the cost function of each one of them.
		 */
		for(int i = 0; i < height; i++){
			numRootsLeft = 0;
			numRootsRight = 0;

			leftRoots = FindLevelRoots(&numRootsLeft, treeLeft, width, i);
			rightRoots = FindLevelRoots(&numRootsRight, treeRight, width, i);

			if(i==0)
				for(int jj = 0; jj < numRootsLeft; jj++)
					printf("Radice N°%d -> %d\n", leftRoots[jj].nodeID, leftRoots[jj].graylevel);


			if(VERBOSE){
				printf("Scanline N°%d\n",i);
				printf("The left tree has %d levelroots\n", numRootsLeft);
				printf("The right tree has %d levelroots\n", numRootsRight);
			}

			rightIndex = 0;
			for(int j = 0; j < numRootsLeft; j++){
				idLeftNode = leftRoots[j].nodeID;

				/**
				 * Search for a root node that is near to the one that is under inspection.
				 */
				while(rightIndex <= numRootsRight && rightRoots[rightIndex].nodeID <= idLeftNode)
					rightIndex++;

				/**
				 * The variable rightIndex now points to the next position to inspect
				 * in the rightRoots array.
				 * This position has a nodeID that is greater of the one on the left
				 * side, for this reason as first match I should take the previous one.
				 */
				currentColorCost = (double)costFunctionColorRoots(treeLeft, treeRight, leftRoots,
													rightRoots, j, (rightIndex-1), i, width,
													numRootsLeft, numRootsRight);
				currentAreaCost = (double)costFunctionAreaRoots(treeLeft, treeRight, leftRoots,
													rightRoots, j, (rightIndex-1), i, width,
													numRootsLeft, numRootsRight);

				/**
				 * The first cost is computed, now try to find a better matching
				 * searching in the previous level roots.
				 * The max depth of this search is defined by the maxShift variable.
				 * If the id found on right is too near to the left side, all possible
				 * matching are computed.
				 */
				numCandidates = ((rightIndex-1-maxShift) < 0) ? rightIndex-1 : maxShift;
				for(int k = 1; k <= numCandidates; k++){
					tmpColorCost = (double)costFunctionColorRoots(treeLeft, treeRight, leftRoots,
														rightRoots, j, (rightIndex-1-k), i, width,
														numRootsLeft, numRootsRight);
					tmpAreaCost = (double)costFunctionAreaRoots(treeLeft, treeRight, leftRoots,
														rightRoots, j, (rightIndex-1-k), i, width,
														numRootsLeft, numRootsRight);

					if(tmpColorCost < currentColorCost && tmpAreaCost < currentAreaCost){
						currentColorCost = tmpColorCost;
						currentAreaCost = tmpAreaCost;
					}
				}

				/**
				 * After this process I have the minimum cost inside the new tree
				 * structure.
				 */
				costColor[idLeftNode + i*width] = currentColorCost;
				costArea[idLeftNode + i*width] = currentAreaCost;
			}

			free(leftRoots);
			free(rightRoots);
		}

		maxAreaCost = 0;
		maxColorCost = 0;
		for(int i = 0; i < height; i++)
			for(int j = 0; j < width; j++)
				if(treeout[j + i*width].levelroot){
					maxAreaCost = (costArea[j + i*width] > maxAreaCost) ?
												costArea[j + i*width]: maxAreaCost;
					maxColorCost = (costColor[j + i*width] > maxColorCost) ?
												costColor[j + i*width]: maxColorCost;
				}

		printf("Max Area Cost: %f \n", maxAreaCost);
		printf("Max Color Cost: %f \n", maxColorCost);

		/* Normalization of the cost */
		for(int i = 0; i < height; i++)
			for(int j = 0; j < width; j++)
				if(treeout[j + i*width].levelroot)
					treeout[j + i*width].gval = (int)(((costArea[j + i*width]/maxAreaCost +
													costColor[j + i*width]/maxColorCost)/2)*255);

		/* Assign cost to the non level root nodes */
		parentRootId = 0;
		for(int i = 0; i < height; i++)
			for(int j = 0; j < width; j++)
				if(!treeout[j + i*width].levelroot){
					parentRootId = treeout[j + i*width].parent;
					while(!treeout[parentRootId].levelroot)
						parentRootId = treeout[parentRootId].parent;
					treeout[j + i*width].gval = treeout[parentRootId].gval;
				}

		return treeout;
}

int costFunctionRoots(MaxNode* treeLeft, MaxNode* treeRight, LevelRoot* leftRoots,
								LevelRoot* rightRoots, int leftIndex, int rightIndex,
								int i, int width, int numleftRoots, int numrightRoots){
	int diffArea, diffParentArea, diffChildArea;
	int diffColor, diffParentColor, diffChildColor;
	//int leftChildIndex, rightChildIndex;
	//int leftParentIndex, rightParentIndex;

	diffArea = computeAreaDiff(treeLeft, treeRight, leftRoots[leftIndex].nodeID + i*width,
									 	 	 rightRoots[rightIndex].nodeID + i*width, width);
	diffColor = computeColorDiff(treeLeft, treeRight, leftRoots[leftIndex].nodeID + i*width,
										     rightRoots[rightIndex].nodeID + i*width);

/*	leftChildIndex = leftIndex-1;
	while(leftIndex < (numleftRoots-1) &&
			treeLeft[leftRoots[leftChildIndex].nodeID + i*width].gval == treeLeft[leftRoots[leftIndex].nodeID + i*width].gval)
		leftChildIndex--;

	rightChildIndex = rightIndex-1;
	while(rightIndex < (numrightRoots-1) &&
			treeRight[rightRoots[rightChildIndex].nodeID + i*width].gval == treeRight[rightRoots[rightIndex].nodeID + i*width].gval)
		rightChildIndex--;


	if(leftIndex <= 0 && rightIndex <= 0){
		diffChildArea = 0;
		diffChildColor = 0;
	} else if(leftIndex <= 0) {
		diffChildArea = width - treeRight[rightRoots[rightChildIndex].nodeID + i*width].Area;
		diffChildColor = 255 - treeRight[rightRoots[rightChildIndex].nodeID + i*width].gval;
	} else if(rightIndex <= 0) {
		diffChildArea = width - treeLeft[leftRoots[leftChildIndex].nodeID + i*width].Area;
		diffChildColor = 255 - treeLeft[leftRoots[leftChildIndex].nodeID + i*width].gval;
	} else {
		diffChildArea = computeAreaDiff(treeLeft, treeRight, leftRoots[leftChildIndex].nodeID + i*width,
												rightRoots[rightChildIndex].nodeID + i*width, width);
		diffChildColor = computeColorDiff(treeLeft, treeRight, leftRoots[leftChildIndex].nodeID + i*width,
												rightRoots[rightChildIndex].nodeID + i*width);
	}



	if(leftIndex >= (numleftRoots-1) && rightIndex >= (numrightRoots-1)){
		diffParentArea = 0;
		diffParentColor = 0;
	} else if(leftIndex >= (numleftRoots-1)){
		diffParentArea = width - treeRight[rightRoots[rightIndex+1].nodeID + i*width].Area;
		diffParentColor = 255 - treeRight[rightRoots[rightIndex+1].nodeID + i*width].gval;
	} else if(rightIndex >= (numrightRoots-1)){
		diffParentArea = width - treeLeft[leftRoots[leftIndex+1].nodeID + i*width].Area;
		diffParentColor = 255 - treeLeft[leftRoots[leftIndex+1].nodeID + i*width].gval;
	} else {
		diffParentArea = computeAreaDiff(treeLeft, treeRight, leftRoots[leftIndex+1].nodeID + i*width,
												rightRoots[rightIndex+1].nodeID + i*width, width);
		diffParentColor = computeColorDiff(treeLeft, treeRight, leftRoots[leftIndex+1].nodeID + i*width,
			     	 	 	 	 	 	 	 	 rightRoots[rightIndex+1].nodeID + i*width);
	}
	*/

	if(leftIndex >= (numleftRoots-1) && rightIndex >= (numrightRoots-1)){
		diffParentArea = 0;
		diffParentColor = 0;
	} else if(leftIndex >= (numleftRoots-1)){
		diffParentArea = width - treeRight[rightRoots[rightIndex+1].nodeID + i*width].Area;
		diffParentColor = 255 - treeRight[rightRoots[rightIndex+1].nodeID + i*width].gval;
	} else if(rightIndex >= (numrightRoots-1)){
		diffParentArea = width - treeLeft[leftRoots[leftIndex+1].nodeID + i*width].Area;
		diffParentColor = 255 - treeLeft[leftRoots[leftIndex+1].nodeID + i*width].gval;
	} else {
		diffParentArea = computeAreaDiff(treeLeft, treeRight, leftRoots[leftIndex+1].nodeID + i*width,
												rightRoots[rightIndex+1].nodeID + i*width, width);
		diffParentColor = computeColorDiff(treeLeft, treeRight, leftRoots[leftIndex+1].nodeID + i*width,
			     	 	 	 	 	 	 	 	 rightRoots[rightIndex+1].nodeID + i*width);
	}

	if(leftIndex <= 0 && rightIndex <= 0){
		diffChildArea = 0;
		diffChildColor = 0;
	} else if(leftIndex <= 0) {
		diffChildArea = width - treeRight[rightRoots[rightIndex-1].nodeID + i*width].Area;
		diffChildColor = 255 - treeRight[rightRoots[rightIndex-1].nodeID + i*width].gval;
	} else if(rightIndex <= 0) {
		diffChildArea = width - treeLeft[leftRoots[leftIndex-1].nodeID + i*width].Area;
		diffChildColor = 255 - treeLeft[leftRoots[leftIndex-1].nodeID + i*width].gval;
	} else {
		diffChildArea = computeAreaDiff(treeLeft, treeRight, leftRoots[leftIndex-1].nodeID + i*width,
												rightRoots[rightIndex-1].nodeID + i*width, width);
		diffChildColor = computeColorDiff(treeLeft, treeRight, leftRoots[leftIndex-1].nodeID + i*width,
												rightRoots[rightIndex-1].nodeID + i*width);
	}

	return diffArea + diffParentArea + diffChildArea + diffColor + diffParentColor + diffChildColor;
}

int costFunctionColorRoots(MaxNode* treeLeft, MaxNode* treeRight, LevelRoot* leftRoots,
									LevelRoot* rightRoots, int leftIndex, int rightIndex,
									int i, int width, int numleftRoots, int numrightRoots){
		int diffColor, diffParentColor, diffChildColor;

		diffColor = computeColorDiff(treeLeft, treeRight,
												leftRoots[leftIndex].nodeID + i*width,
											    rightRoots[rightIndex].nodeID + i*width);

		if(leftIndex >= (numleftRoots-1) && rightIndex >= (numrightRoots-1))
			diffParentColor = 0;
		else if(leftIndex >= (numleftRoots-1))
			diffParentColor = 255 - treeRight[rightRoots[rightIndex+1].nodeID + i*width].gval;
		else if(rightIndex >= (numrightRoots-1))
			diffParentColor = 255 - treeLeft[leftRoots[leftIndex+1].nodeID + i*width].gval;
		else
			diffParentColor = computeColorDiff(treeLeft, treeRight,
													leftRoots[leftIndex+1].nodeID + i*width,
													rightRoots[rightIndex+1].nodeID + i*width);

		if(leftIndex <= 0 && rightIndex <= 0)
			diffChildColor = 0;
		else if(leftIndex <= 0)
			diffChildColor = 255 - treeRight[rightRoots[rightIndex-1].nodeID + i*width].gval;
		else if(rightIndex <= 0)
			diffChildColor = 255 - treeLeft[leftRoots[leftIndex-1].nodeID + i*width].gval;
		else
			diffChildColor = computeColorDiff(treeLeft, treeRight,
													leftRoots[leftIndex-1].nodeID + i*width,
													rightRoots[rightIndex-1].nodeID + i*width);

		return diffColor + diffParentColor + diffChildColor;
}

int costFunctionAreaRoots(MaxNode* treeLeft, MaxNode* treeRight, LevelRoot* leftRoots,
								LevelRoot* rightRoots, int leftIndex, int rightIndex,
								int i, int width, int numleftRoots, int numrightRoots){
	int diffArea, diffParentArea, diffChildArea;

	diffArea = computeAreaDiff(treeLeft, treeRight,
											leftRoots[leftIndex].nodeID + i*width,
									 	 	rightRoots[rightIndex].nodeID + i*width, width);

	if(leftIndex >= (numleftRoots-1) && rightIndex >= (numrightRoots-1))
		diffParentArea = 0;
	else if(leftIndex >= (numleftRoots-1))
		diffParentArea = width - treeRight[rightRoots[rightIndex+1].nodeID + i*width].Area;
	else if(rightIndex >= (numrightRoots-1))
		diffParentArea = width - treeLeft[leftRoots[leftIndex+1].nodeID + i*width].Area;
	else
		diffParentArea = computeAreaDiff(treeLeft, treeRight,
												leftRoots[leftIndex+1].nodeID + i*width,
												rightRoots[rightIndex+1].nodeID + i*width, width);

	if(leftIndex <= 0 && rightIndex <= 0)
		diffChildArea = 0;
	else if(leftIndex <= 0)
		diffChildArea = width - treeRight[rightRoots[rightIndex-1].nodeID + i*width].Area;
	else if(rightIndex <= 0)
		diffChildArea = width - treeLeft[leftRoots[leftIndex-1].nodeID + i*width].Area;
	else
		diffChildArea = computeAreaDiff(treeLeft, treeRight,
												leftRoots[leftIndex-1].nodeID + i*width,
												rightRoots[rightIndex-1].nodeID + i*width, width);

	return diffArea + diffParentArea + diffChildArea;
}

int computeAreaDiff(MaxNode* treeLeft, MaxNode* treeRight, int idLeftNode,
												int idRightNode, int width){
	int areaLeft;
	int areaRight;

	areaLeft = (idLeftNode == -1) ? width : treeLeft[idLeftNode].Area;
	areaRight = (idRightNode == -1) ? width : treeRight[idRightNode].Area;

	return abs((areaLeft-areaRight));
}

int computeColorDiff(MaxNode* treeLeft, MaxNode* treeRight, int idLeftNode, int idRightNode){
	if(idLeftNode == -1 || idRightNode == -1)
		return 0;
	return abs(treeLeft[idLeftNode].gval - treeRight[idRightNode].gval);
}
