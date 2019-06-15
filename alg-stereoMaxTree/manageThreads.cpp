/**
 * Max Tree Stereo Matching
 * manageThreads.c
 *
 * Module of the program that manages the workload to assign to each thread in
 * order to parallelize the execution of the program.
 *
 */

#include "include/manageThreads.hpp"

/**
 * This function builds all max trees of an image dividing the workload on a chosen number of threads.
 * @param	nthreads	Total number of threads used by the program
 * @param	imgfname	Filename of the input image
 * @param	outfname	Filename of the output image
 * @param	node		Pointer to the maxtree to create
 * @param	height		Height of the image
 * @param	width		Width of the image
 * @return				Returns a pointer to the max tree built
 */
MaxNode* BuildMaxTrees(int nthreads, const char* imgfname, MaxNode *node, int *height,
														int *width, bool maxtree) {
	Pixel *img = NULL;
	int size;
	ThreadData *thdata;
	pthread_t threadID[MAXTHREADS];

	FreeImage_Initialise(0);
	img = ReadTIFF(imgfname, width, height);
	if (!img){
		printf("ReadTIFF returned null");
		return NULL;
	}
	size = (*width)*(*height);

	if(VERBOSE){
		printf("Processing image '%s'\n", imgfname);
		printf("Image: Width=%d Height=%d Size=%d\n", *width, *height, size);
		printf("Number of threads: %d\n", nthreads);
	}

	node = (MaxNode*) calloc((size_t)size, sizeof(MaxNode));
	if (node==NULL) {
		fprintf(stderr, "out of memory! \n");
		free(img);
		return NULL;
	}

	thdata = MakeThreadData(nthreads, img, *width, *height, node, maxtree);
	if(VERBOSE)
		printf("Data read, start processing.\n");

	BuildTree(thdata, nthreads, threadID);

	free(img); // remi

	free(thdata);
	FreeImage_DeInitialise();
	return node;
}

/**
 * This function makes data for all threads that will be used in the program.
 * @param	nthreads	Total number of threads used in the program
 * @param	gval		Pointer to the array of pixels of the image
 * @param	width		Width of the image
 * @param	height		Height of the image
 * @param	depth		Depth of the image
 * @param	size2D		Size 2D of the image
 * @param	node		Pointer to the max tree on which threads should work
 * @return				Returns a pointer to an array of data for threads
 */
ThreadData *MakeThreadData(int nthreads, Pixel* gval, int width,
								int height, MaxNode* node, bool maxtree){

  ThreadData *data = (ThreadData*) malloc(nthreads *sizeof(ThreadData));
  for (int i=0; i < nthreads; i++){
    data[i].self = i;
    data[i].gval = gval;
    data[i].node = node;
    data[i].height = height;
    data[i].width = width;
    data[i].nthreads = nthreads;
    data[i].maxtree = maxtree;
    if(VERBOSE){
    	printf("Thread %d, Lower bound = %d, Upper bound = %d\n",i,LWB(i),UPB(i));
    	printf("width: %d - height: %d\n", width, height);
    }
  }
  return data;
}

/**
 * This function, given threads and their data let the threads start.
 * @param	thdata		Pointer to an array of data for threads
 * @param	nthreads	Total number of threads used into the program
 * @param	threadID	Array of threads
 */
void BuildTree(ThreadData *thdata, int nthreads, pthread_t threadID[]) {
  int thread;
  for (thread=0; thread < nthreads; ++thread) {
    pthread_create(threadID+thread, NULL, ThreadRun, (void *) (thdata + thread));
  }
  for (thread=0; thread < nthreads; ++thread) {
    pthread_join(threadID[thread], NULL);
  }
}

/**
 * This function represent the work done by each thread. In particular it invokes
 * the right method to build and to filter a max tree.
 * @param 	arg		Pointer to the area of memory where arguments of the thread
 * 					are stored
 */
void *ThreadRun(void *arg) {
  ThreadData *thdata = (ThreadData *) arg;
  int self, width, height, nthreads;
  bool maxtree;
  Pixel *gval;
  MaxNode *node;

  /**
   * Getting arguments from thread data
   */
  self = thdata->self;
  width = thdata->width;
  height = thdata->height;
  gval = thdata->gval;
  node = thdata->node;
  nthreads = thdata->nthreads;
  maxtree = thdata->maxtree;

  BuildOpThread(self, gval, width, height, nthreads, node, maxtree);

  return NULL;
}

/**
 * This function takes care of the creation of the max trees for a single
 * thread invoking the correct function for each scan line of its part of
 * the image.
 * @param	thread		Number of the Thread
 * @param	gval		Pointer to the array of pixels of the whole image
 * @param	width		Width of the image
 * @param	height		Height of the image
 * @param	depth		Depth of the image
 * @param	nthreads	Number of total threads used
 * @param	node		Maxtree created
 */
void BuildOpThread(int thread, Pixel *gval, int width, int height,
								int nthreads, MaxNode *node, bool maxtree){
	int y, lwby, upby;

	lwby = (thread*height)/nthreads;
	upby = ((thread+1)*height)/nthreads;

	for (y = lwby; y < upby; y++)
		if(maxtree)
			BuildMaxTree1D(y, gval, width, height, node);
		else
			BuildMinTree1D(y, gval, width, height, node);
}

/**
 * Reads the memory usage of the program from the output of the /proc/self/status
 * command and stores useful values in relative variables.
 * In particular:
 * 		VmPeak: Peak virtual memory size
 * 		VmSize: Virtual memory size
 * 		VmHWM: Peak resident set size
 * 		VmRSS: Resident set size
 * @param	phy_mem		Current Physical memory used
 * @param	mphy_mem	Maximum Physical memory used
 * @param	virt_mem	Current Virtual memory used
 * @param	mvirt_mem	Maximum Virtual memory used
 */
void getMemory(int* phy_mem, int* mphy_mem, int* virt_mem, int* mvirt_mem) {
	char* curr_phy_mem;
	char* max_phy_mem;
	char* curr_virt_mem;
	char* max_virt_mem;
	char* line;
	size_t len;
	FILE* file;

	curr_phy_mem = NULL;
	curr_virt_mem = NULL;
	max_phy_mem = NULL;
	max_virt_mem = NULL;
	len = 128;
	line = (char*)malloc(128);

	file = fopen("/proc/self/status", "r");
	if(!file)
		exit(1);

	/*
	 * While at least one of the values is not found.
	 */
	while(!curr_phy_mem || !max_phy_mem || !curr_virt_mem || !max_virt_mem){
		/* File ends */
		if(getline(&line, &len, file) == -1){
			exit(1);
		}

		/* Find current physical memory */
		if(!strncmp(line, "VmRSS:", 6)){
			curr_phy_mem = strdup(&line[7]);
		}

		/* Find max physical memory */
		if(!strncmp(line, "VmHWM:", 6)){
			max_phy_mem = strdup(&line[7]);
		}

		/* Find current virtual memory */
		if(!strncmp(line, "VmSize:", 7)){
			curr_virt_mem = strdup(&line[7]);
		}

		/* Find max virtual memory */
		if(!strncmp(line, "VmPeak:", 7)){
			max_virt_mem = strdup(&line[7]);
		}
	}

	/*
	 * Removes the "Kb" string in order to convert it to an int
	 * value.
	 */
	len = strlen(curr_phy_mem);
	curr_phy_mem[len - 4] = 0;
	len = strlen(max_phy_mem);
	max_phy_mem[len - 4] = 0;
	len = strlen(curr_virt_mem);
	curr_virt_mem[len - 4] = 0;
	len = strlen(max_virt_mem);
	max_virt_mem[len - 4] = 0;

	*phy_mem = atoi(curr_phy_mem);
	*mphy_mem = atoi(max_phy_mem);
	*virt_mem = atoi(curr_virt_mem);
	*mvirt_mem = atoi(max_virt_mem);

	free(line);
	free(curr_phy_mem);
	free(max_phy_mem);
	free(curr_virt_mem);
	free(max_virt_mem);
	fclose(file);
}

double getWallTime(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

