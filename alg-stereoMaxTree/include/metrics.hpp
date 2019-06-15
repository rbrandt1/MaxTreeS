
#ifndef INCLUDE_METRICS_HPP_
#define INCLUDE_METRICS_HPP_

#include "config.hpp"

float metric_bad(cv::Mat groundTruth,cv::Mat prediction,float error);
float metric_avgerr(cv::Mat groundTruth,cv::Mat prediction);
float metric_rms(cv::Mat groundTruth,cv::Mat prediction);
float metric_percentile(cv::Mat groundTruth,cv::Mat prediction,float percentile);
void calcMetrics(const cv::Mat& dispMap, int disparityLevelsResized,
		double endTotal, double startTotal,std::string folder,int divider);

#endif /* INCLUDE_METRICS_HPP_ */
