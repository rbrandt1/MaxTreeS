/**
 * Max Tree Stereo Matching
 * config.h
 *
 * Configuration file of the program. It holds all constants, and some
 * configurations of the program.
 *
 * @author: A.Fortino
 */

#ifndef INCLUDE_CONFIG_HPP_
#define INCLUDE_CONFIG_HPP_

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


// Max number of threads in the program
#define MAXTHREADS 			128
// Name of the output image file
#define OUT_IMAGE			"outputImage.png"
// Name of the output text file
#define FILE_TEXT_OUT_NAME	"outputText.txt"
#define FILE_IMG_DISP		"disp.png"
// Max number of pixel to inspect on left for searching a match
#define MAX_SHIFT			10
// Variable that enables more output in console
#define	VERBOSE				0
#define DEBUG				0
#define SAVE_DISP_AS_PNG	1
#define MIN_TREE_BUILDING	1
#define CONVERT_TO_16B		0
// Value that indicates a root
#define ROOT				(-1)
// Default quantization levels
#define QUANT_LEVELS		256
#define MAX_DISPLACEMENT	2
#define MEDIAN_KERNEL_SIZE	11
// Definition of the bool type, with true and false values
#define false 0
#define true  1
// Definition of the unsigned byte and Pixel types
typedef unsigned char ubyte;
typedef unsigned char Pixel;

// Definition of some useful macros
#define LWB(self) ((width*height)*(((self)*height)/nthreads))
#define UPB(self) ((width*height)*(((self+1)*height)/nthreads))

// Definition of the Max Tree Node structure
typedef struct MaxNode	{
  int parent;
  int Area;
  Pixel gval;
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

#endif /* INCLUDE_CONFIG_HPP_ */
