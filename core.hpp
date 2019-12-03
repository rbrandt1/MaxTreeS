#ifndef MAIN_H
#define MAIN_H


#include <cmath>
#include <math.h>
#include <iostream>
#include <numeric>
#include <queue>
#include "opencv_pfm.hpp"
#include <iostream>
#include <fstream>
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
#include <chrono>
#include <time.h>
#include <sys/time.h>

/****** Definitions ******/

#define ROOT				(-1)

// Type definition of a Max-Tree node


void writeLine(std::string filename,std::string line);
void clearFile(std::string filename);
double getWallTime();
void writeVisDispmap(cv::Mat img, std::string str, float dispLevels);


cv::Mat work(cv::Mat imgLeft, cv::Mat imgRight,bool sparse,float alpha,int minAreaMatchedFineTopNode,int maxAreaMatchedFineTopNode,int nColors,int * sizes, int total,uint kernelCostComp,int kernelCostVolume, int everyNthRow, int disparityLevels,float coonfidence,float allowanceSearchRange);


#endif 

