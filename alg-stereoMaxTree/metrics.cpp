
#include "include/metrics.hpp"
#include "opencv_pfm.hpp"

/* metrics */


float metric_bad(cv::Mat groundTruth,cv::Mat prediction,float error){
	long badPixels = 0;
	long badPixelsError = 0;

	for(int r = 0; r < prediction.rows; r++){
		for(int c = 0; c < prediction.cols; c++){

			uchar groundTruthPixel = groundTruth.at<uchar>(r,c);
			uchar predictionPixel = prediction.at<uchar>(r,c);
			if(0 != groundTruthPixel-predictionPixel){
				if(error < abs((int)groundTruthPixel-(int)predictionPixel)){
					badPixelsError++;
				}
				badPixels++;
			}
		}
	}
	if(badPixels == 0){
		return 0;
	}
	return ((float)badPixelsError)/((float)badPixels)*100;
}

float metric_avgerr(cv::Mat groundTruth,cv::Mat prediction){
	std::vector<float> list;

	for(int r = 0; r < prediction.rows; r++){
		for(int c = 0; c < prediction.cols; c++){
			ushort groundTruthPixel = groundTruth.at<ushort>(r,c);
			ushort predictionPixel = prediction.at<ushort>(r,c);
			if(groundTruthPixel != 0 && predictionPixel != 0){ // !!!!!!!!
				list.push_back(abs(groundTruthPixel-predictionPixel));
			}
		}
	}

	std::vector<float> mean;
	cv::reduce(list,mean,1,cv::REDUCE_AVG);

	return mean[0];
}

float metric_rms(cv::Mat groundTruth,cv::Mat prediction){
	std::vector<float> list;
	for(int r = 0; r < prediction.rows; r++){
		for(int c = 0; c < prediction.cols; c++){
			ushort groundTruthPixel = groundTruth.at<ushort>(r,c);
			ushort predictionPixel = prediction.at<ushort>(r,c);
			if(groundTruthPixel != 0 && predictionPixel != 0){ // !!!!!!!!
				int a = abs(groundTruthPixel-predictionPixel);
				list.push_back(a*a);
			}
		}
	}
	std::vector<float> mean;
	cv::reduce(list,mean,01,cv::REDUCE_AVG);
	return std::sqrt(mean[0]);
}



float metric_percentile(cv::Mat groundTruth,cv::Mat prediction,float percentile){
	std::vector<float> list;
	for(int r = 0; r < prediction.rows; r++){
		for(int c = 0; c < prediction.cols; c++){
			uchar groundTruthPixel = groundTruth.at<uchar>(r,c);
			uchar predictionPixel = prediction.at<uchar>(r,c);
			if(0 != groundTruthPixel-predictionPixel){
				list.push_back(abs(groundTruthPixel-predictionPixel));
			}
		}
	}
	std::sort(list.begin(),list.end(),
			[](const float &a,const float &b){
					return a < b;
				});
	int index = percentile * (list.end()-list.begin());
	return list.at(index);
}



void calcMetrics(const cv::Mat& dispMap, int disparityLevels,
		double endTotal, double startTotal,std::string folder,int divider) {
	// Metrics
	printf("-----------------------------\n");
	std::string namegt = folder;
	namegt.append("/").append("disp1.pfm");
	cv::Mat groundTruth = opencv_pfm::imread_pfm(namegt.c_str(), -1);

	ushort inf = std::numeric_limits<ushort>::infinity();
	ushort max2 = std::numeric_limits<ushort>::max();



	cv::resize(groundTruth, groundTruth, cv::Size(dispMap.cols, dispMap.rows));

	groundTruth.convertTo(groundTruth,CV_16U);

	for(int r =0;r<groundTruth.rows ; r++){
		for(int c =0;c<groundTruth.cols ; c++){
			if(groundTruth.at<ushort>(r,c) == inf || groundTruth.at<ushort>(r,c) == max2){
				groundTruth.at<ushort>(r,c) =0;
			}
		}
	}

	groundTruth /= 256;

	cv::Mat error = cv::Mat(groundTruth.rows,groundTruth.cols,CV_16U);
	float scale = 10;
	for(int r =0;r<dispMap.rows ; r++){
		for(int c =0;c<dispMap.cols ; c++){
			float groundTruthPixel = groundTruth.at<ushort>(r,c);
			ushort predictionPixel = dispMap.at<ushort>(r,c);
			if(groundTruthPixel != 0 && predictionPixel != 0){
				error.at<ushort>(r,c) =  scale * abs(groundTruth.at<ushort>(r,c) - dispMap.at<ushort>(r,c));
			}else{
				error.at<ushort>(r,c) =0;
			}
		}
	}
	cv::Mat dispMapcm;
	cv::Mat dispMapFinal;
	dispMapFinal = error;
	dispMapFinal.convertTo(dispMapFinal, CV_8U);
	cv::applyColorMap(dispMapFinal, dispMapcm, cv::COLORMAP_JET);
	cv::imwrite("error.png", dispMapcm);



	std::vector<ushort> max;
	cv::reduce(groundTruth,max,01,cv::REDUCE_MAX);
	int maxGT = max[0];

	std::vector<float> mean;
	cv::reduce(groundTruth,mean,01,cv::REDUCE_AVG);
	float meanGT = mean[0];

	mean.clear();
	cv::reduce(dispMap,mean,01,cv::REDUCE_AVG);
	float meanDisp = mean[0];

	std::vector<ushort> max22;
	cv::reduce(dispMap,max22,01,cv::REDUCE_MAX);
	int maxDisp = max22[0];

	printf("meanDisp %f |  meanGT %f | maxDisp %d | maxGT %d\n",meanDisp,meanGT,maxDisp,maxGT);

	float MP = (groundTruth.rows / 1000.0) * (groundTruth.cols / 1000.0);
	float metric;
	//	metric = metric_bad(groundTruth,dispMap,.5);
	//	printf("bad 0.5 = %f \n",metric);
	//	metric = metric_bad(groundTruth,dispMap,1);
	//	printf("bad 1.0 = %f \n",metric);
	//	metric = metric_bad(groundTruth,dispMap,2);
	//	printf("bad 2.0 = %f \n",metric);
	//	metric = metric_bad(groundTruth,dispMap,4);
	//	printf("bad 4.0 = %f \n",metric);
	metric = metric_avgerr(groundTruth, dispMap);
	printf("avgerr = %f \n", metric);
	metric = metric_rms(groundTruth, dispMap);
	printf("rms = %f \n", metric);
	//	metric = metric_percentile(groundTruth,dispMap,.5);
	//	printf("percentile 0.5 = %f \n",metric);
	//	metric = metric_percentile(groundTruth,dispMap,.9);
	//	printf("percentile 0.90 = %f \n",metric);
	//	metric = metric_percentile(groundTruth,dispMap,.95);
	//	printf("percentile 0.95 = %f \n",metric);
	//	metric = metric_percentile(groundTruth,dispMap,.99);
	//	printf("percentile 0.99 = %f \n",metric);
	printf("Time/MP %.4f \n", (endTotal - startTotal) / MP);
}
