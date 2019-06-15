/**
 * Max Tree Stereo Matching
 * maxtree.c
 *
 * Max Tree module of the program. This module implements all functions needed to
 * build max tree and identify level roots inside the trees.
 *
 * @author: A.Fortino
 */

#include "include/maxtree.hpp"

/**
 * This function builds a max tree for a single scanline.
 * @param	row		Number that identifies the scanline
 * @param	gval	Pointer to the array of pixels from which build the max tree
 * @param	width	Width of the image
 * @param	height	Height of the image
 * @param	depth	Depth of the image
 * @param	node	Maxtree created
 */
void BuildMaxTree1D(int row, Pixel *gval, int width, int height, MaxNode *node){
	int i, current, curr_parent, next;

	/* Create singleton max-trees
	 * A max-node is created for each pixel in the image. All nodes are
	 * initialized as roots, then they have a pointer to -1 (no meaning
	 * value).
	 */
	for (i = 0, current = row * width; i < width; i++, current++){
		node[current].parent = ROOT;
		node[current].gval = gval[current];
		node[current].Area = 1;
		node[current].levelroot = true;
		node[current].nChildren = 0;
		node[current].begIndex = std::numeric_limits<int>::max();
		node[current].dispL = -1;
		node[current].dispR = -1;
		node[current].matchId = -1;
		node[current].curLevel = false;
		node[current].prevCurLevel = false;
	//	node[current].totalNChildren = 0;
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
		if (gval[current] <= gval[next]) { /* ascending or flat */

			node[next].parent = current;

			if (gval[current] == gval[next]){ /*flat */
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
			while ((curr_parent!=ROOT) && (gval[curr_parent]>gval[next])){
				node[curr_parent].Area += node[current].Area;
				current = curr_parent;
				curr_parent = node[current].parent;
			}
			node[current].parent = next;
			if(gval[current] == gval[node[current].parent] && next != ROOT)
				node[current].levelroot = false;
			node[next].Area += node[current].Area;
			node[next].parent = curr_parent;
			if(gval[node[next].parent] == gval[next] && curr_parent != ROOT)
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



	/*
	for (i = row * width; i < (row+1) * width; i++){
			int current = i;
			if(node[current].levelroot && node[current].nChildren == 0 ){

				while(node[current].parent != ROOT){
					node[current].totalNChildren++;
					current = node[current].parent;
				}
			}
		}
*/




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


/**
 * Function that builds an array of level roots ordered from higher to lower.
 * @param	size	Output param containing the size of the array of level roots
 * @param	tree	Maxtree of which find level roots
 * @param	width	Width of the image
 * @param	row		Number of the scanline of the image
 * @return			Array of level roots ordered
 */
LevelRoot* FindLevelRoots(int *size, MaxNode *tree, int width, int row){
	LevelRoot *roots = NULL;
	for (int i = 0, currentNode = row * width; i < width; i++, currentNode++)
		if(tree[currentNode].levelroot)
			roots = InsertRootInOrder(roots, size, tree[currentNode].gval, i);

	return roots;
}

/**
 * Function that performs an in order insertion in the array of level roots.
 * @param	roots		Array of level roots
 * @param	size 		Size of the array of level roots
 * @param	graylevel	Graylevel of the level root to insert
 * @param	nodeID		Node to insert as level root
 * @return				Array of level roots ordered
 */
LevelRoot* InsertRootInOrder(LevelRoot* roots, int *size, int graylevel, int nodeID){
	LevelRoot *new_roots = NULL;

	(*size)++;
	new_roots = (LevelRoot*)realloc(roots, (*size)*sizeof(LevelRoot));
	if(new_roots == NULL){
		if(DEBUG)
			printf("Error in reallocation of the array of LevelRoots!\n");
		exit(0);
	} else if (new_roots != roots) {
		if(DEBUG)
			printf("Reallocation completed!\n");
		roots = new_roots;
	}
	new_roots = NULL;
	free(new_roots);

	roots[(*size)-1].graylevel = graylevel;
	roots[(*size)-1].nodeID = nodeID;
	return roots;
}

void sortLevelRoots(LevelRoot* roots, int low, int high){
	if(low < high){
		int index = partition(roots, low, high);
		sortLevelRoots(roots, low, index-1);
		sortLevelRoots(roots, index+1, high);
	}
}

int partition(LevelRoot* roots, int low, int high){
	int pivot = roots[high].graylevel;
	int min_index = low-1;
	LevelRoot tmp;

	for(int i = low; i <= high-1; i++)
		if(roots[i].graylevel <= pivot){
			min_index++;
			tmp.graylevel = roots[i].graylevel;
			tmp.nodeID = roots[i].nodeID;
			roots[i].graylevel = roots[min_index].graylevel;
			roots[i].nodeID = roots[min_index].nodeID;
			roots[min_index].graylevel = tmp.graylevel;
			roots[min_index].nodeID = tmp.nodeID;
		}

	tmp.graylevel = roots[min_index+1].graylevel;
	tmp.nodeID = roots[min_index+1].nodeID;
	roots[min_index+1].graylevel = roots[high].graylevel;
	roots[min_index+1].nodeID = roots[high].nodeID;
	roots[high].graylevel = tmp.graylevel;
	roots[high].nodeID = tmp.nodeID;

	return min_index+1;
}

/**
 * This function builds a max tree for a single scanline.
 * @param	row		Number that identifies the scanline
 * @param	gval	Pointer to the array of pixels from which build the max tree
 * @param	width	Width of the image
 * @param	height	Height of the image
 * @param	depth	Depth of the image
 * @param	node	Maxtree created
 */
void BuildMinTree1D(int row, Pixel *gval, int width, int height, MaxNode *node){
	int i, current, curr_parent, next;

	/* Create singleton max-trees
	 * A max-node is created for each pixel in the image. All nodes are
	 * initialized as roots, then they have a pointer to -1 (no meaning
	 * value).
	 */
	for (i = 0, current = row * width; i < width; i++, current++){
		node[current].parent = ROOT;
		node[current].gval = gval[current];
		node[current].Area = 1;
		node[current].levelroot = true;
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
		if (gval[current] >= gval[next]) { /* ascending or flat */
			node[next].parent = current;
			if (gval[current] == gval[next]){ /*flat */
				node[current].Area ++;
				node[next].levelroot = false;
			} else {
				// node[current].children.push_back(next);
				current = next;  /* new top level root */
			}
		} else { /* descending */
			curr_parent = node[current].parent; // save the current node parent
			/*
			 * For each parent of the current node, until a root node is reached or
			 * when the gray value of the current node becomes lower, this procedure
			 * of swapping parents should be repeated.
			 */
			while ((curr_parent!=ROOT) && (gval[curr_parent]<gval[next])){
				node[curr_parent].Area += node[current].Area;
				current = curr_parent;
				curr_parent = node[current].parent;
			}
			node[current].parent = next;
			if(gval[current] == gval[node[current].parent] && next != ROOT)
				node[current].levelroot = false;
			node[next].Area += node[current].Area;
			node[next].parent = curr_parent;
			if(gval[node[next].parent] == gval[next] && curr_parent != ROOT)
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
}
