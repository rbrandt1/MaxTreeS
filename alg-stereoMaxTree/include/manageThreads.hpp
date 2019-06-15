/**
 * Max Tree Stereo Matching
 * manageThreads.h
 *
 * Module of the program that manages the workload to assign to each thread in
 * order to parallelize the execution of the program.
 *
 * @author: A.Fortino
 */

#ifndef INCLUDE_MANAGETHREADS_HPP_
#define INCLUDE_MANAGETHREADS_HPP_

#include <pthread.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/stat.h>
#include <unistd.h>
#include "config.hpp"
#include "manageImages.hpp"
#include "maxtree.hpp"

/**
 * Data structure that represent all data needed from a Thread for running
 */
typedef struct {
	int self;
	Pixel* gval;
	int width;
	int height;
	int nthreads;
	MaxNode *node;
	bool maxtree;
} ThreadData;

MaxNode* BuildMaxTrees(int nthreads, const char* imgfname, MaxNode *node, int *height,
														int *width, bool maxtree);
ThreadData *MakeThreadData(int nthreads, Pixel* gval, int width, int height,
														MaxNode* node, bool maxtree);
void BuildTree(ThreadData *thdata, int nthreads, pthread_t threadID[]);
void *ThreadRun(void *arg);
void BuildOpThread(int thread, Pixel *gval, int width, int height, int nthreads,
														MaxNode *node, bool maxtree);
void getMemory(int* phy_mem, int* mphy_mem, int* virt_mem, int* mvirt_mem);
double getWallTime();

#endif /* INCLUDE_MANAGETHREADS_HPP_ */
