/**
 * Max Tree Stereo Matching
 * main.c
 *
 * Main module of the program. It checks the input parameters and executes
 * the matching algorithm.
 *
 * @author: A.Fortino
 */

#include "include/config.hpp"
#include "include/matching.hpp"

/**
 * Main function of the program.
 */
int main (int argc, char *argv[]){
	int numThreads;
	const char *filenameLeftImage, *filenameRightImage;
	const char *filenameImageOut;
	const char *filenameDisp;
	int maxPixelShift, maxDisplacement, kernelSize;
	int convert, typeVolumeFiltering, typeDisparityFiltering, dispKernelSize;

	if(argc < 4){ // Checks if the minimum number of inputs is reached.
		printf("Usage: %s <num threads> <input image left> <input image right> "
				"[max pixel shift] [max displacement] "
				"[median kernel size] [output image] "
				"[convert to 16bit] [type volume filtering] [type disp filtering] "
				"[kernel disparity filter]\n", argv[0]);
		exit(0);
	}

	numThreads = MIN(atoi(argv[1]), MAXTHREADS);
	filenameLeftImage = argv[2];
	filenameRightImage = argv[3];
	maxDisplacement = (argc > 4) ? atoi(argv[4]) : MAX_DISPLACEMENT;
	const char * testCase = argv[5];

	const char * filename = argv[6];
	kernelSize = (argc > 6) ? atoi(argv[6]) : MEDIAN_KERNEL_SIZE;
	filenameImageOut = (argc > 7) ? argv[7] : OUT_IMAGE;
	convert = (argc > 8) ? atoi(argv[8]) : CONVERT_TO_16B;
	typeVolumeFiltering = (argc > 9) ? atoi(argv[9]) : 1;
	typeDisparityFiltering = (argc > 10) ? atoi(argv[10]) : 0;
	dispKernelSize = (argc > 11) ? atoi(argv[11]) : 5;
	filenameDisp = (argc > 12) ? argv[12] : FILE_IMG_DISP;

	run(numThreads, filenameLeftImage, filenameRightImage, maxPixelShift,
							filenameImageOut, maxDisplacement, kernelSize,
							convert, typeVolumeFiltering, typeDisparityFiltering,
											dispKernelSize, filenameDisp,testCase,filename);



	return 0;
}
