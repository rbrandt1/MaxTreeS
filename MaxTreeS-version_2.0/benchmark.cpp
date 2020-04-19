#include "core.hpp"
#include "opencv_pfm.hpp"
#include <cstdlib>
#include <chrono>
#include <thread>

bool isMetricCompMode;
bool isResultsCompMode;

std::string rootFolder = "./../datasets"; // /mnt/NAS/MaxTreeStereoData

bool ends_with(std::string  const & value, std::string  const & ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void writeDispmap(cv::Mat img, std::string str) {

  for(int r =0;r<img.rows;r++){
	for(int c =0;c<img.cols;c++){
	  if(img.at<float>(r,c) == -1){
		  img.at<float>(r,c) = 0;
	  };
	}
  }
  
  if (ends_with(str, ".pfm")) {
	for(int r =0;r<img.rows;r++){
		for(int c =0;c<img.cols;c++){
		  if(img.at<float>(r,c) == 0){
			  img.at<float>(r,c) = std::numeric_limits<double>::infinity();
		  };
		}
	}
	  
    opencv_pfm::imwrite_pfm(str.c_str(), img,1,1);
  } else {
    cv::imwrite(str.c_str(), img);
  }
}


void writeInfo(std::string method, std::string versionString, double startTotal, cv::Mat img, std::string filename, std::string leftImg_string) {
  double endTotal = getWallTime();
  double totalTime = (endTotal - startTotal);
  double MPS = ((float) img.cols / (float) 1000.0) *
  ((float) img.rows / (float) 1000.0);
  std::string line = method + "," + versionString + "," + leftImg_string + "," + std::to_string(MPS) + "," + std::to_string(totalTime);
  writeLine(filename, line);

  printf("%s\n", line.c_str());
}


void writeInfoResults(std::string filename, std::string methodName, std::string versionString, std::string resultsFile_String, std::string leftImg_string, float avgerr, float rmse, float bad1, float bad2, float bad3, float bad4, float bad5, float d1all,float density,float avgerr_clipped, float rmse_clipped, float bad1_clipped, float bad2_clipped, float bad3_clipped, float bad4_clipped, float bad5_clipped){
  std::string line = methodName + "," + versionString + "," + leftImg_string + "," + std::to_string(avgerr) + "," + std::to_string(rmse) + "," + std::to_string(bad1)+ "," + std::to_string(bad2)+ "," + std::to_string(bad3)+ "," + std::to_string(bad4)+ "," + std::to_string(bad5)+ "," + std::to_string(d1all)+ "," + std::to_string(density)+ "," + std::to_string(avgerr_clipped) + "," + std::to_string(rmse_clipped) + "," + std::to_string(bad1_clipped)+ "," + std::to_string(bad2_clipped)+ "," + std::to_string(bad3_clipped)+ "," + std::to_string(bad4_clipped)+ "," + std::to_string(bad5_clipped);
  writeLine(filename, line);

  printf("%s\n", line.c_str());
}

void initializeResultsFile(std::string filename) {
  clearFile(filename);
  std::string line = "Dataset,Version,LeftImg_string,MPs,Total time";
  writeLine(filename, line);

  printf("%s\n", line.c_str());
}

void initializeResultsAccuracyFile(std::string filename) {
  clearFile(filename);
  std::string line = "Dataset,Version,LeftImg_string,avgerr_clipped,rmse_clipped,bad 1.0_clipped,bad 2.0_clipped,bad 3.0_clipped,bad 4.0_clipped,bad 5.0_clipped,D1(all),density,avgerr,rmse,bad 1.0,bad 2.0,bad 3.0,bad 4.0,bad 5.0";
  writeLine(filename, line);

  printf("%s\n", line.c_str());
}


cv::Mat preProcess(cv::Mat img) {
  if (img.channels() > 1){
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
  }
  
  return img;
}


std::string zeroPadded(std::string old_string, int n_zero) {
  return std::string(n_zero - old_string.length(), '0') + old_string;
}


void mkdir2(std::string dirname){
	std::string command = "mkdir -p "+dirname;
	const int dir_err = system(command.c_str());
	if (-1 == dir_err)
	{
   printf("Error creating directory!n");
   exit(1);
 }
}


bool isValid(float val){
	return val >= 0 && val < 100000;
}


cv::Mat writeVisError(cv::Mat gt, cv::Mat dispMap,int dispLevels){

	if(true){

		  cv::Mat error = dispMap.clone();

		  for (int r = 0; r < dispMap.rows; r++) {
		    for (int c = 0; c < dispMap.cols; c++) {
			if(!isValid(gt.at<float>(r,c)) || !isValid(dispMap.at<float>(r,c))){
				error.at<float>(r,c) = 255;
			} 
			if(!isValid(gt.at<float>(r,c))){
				gt.at<float>(r,c) = 255;
			}
			if(!isValid(dispMap.at<float>(r,c))){
				dispMap.at<float>(r,c) = 255;
			} 

		    }
		  }
		  
		  gt = gt / dispLevels * 255;
		  dispMap = dispMap / dispLevels * 255;
		  error = error / dispLevels * 255;

		  cv::Mat img;
		   
		  gt.convertTo(img, CV_8U);
		  cv::applyColorMap(img, img, cv::COLORMAP_JET);
		  cv::imwrite("error1.png", img);

		  dispMap.convertTo(img, CV_8U);
		  cv::applyColorMap(img, img, cv::COLORMAP_JET);
		  cv::imwrite("error2.png", img);

		  error.convertTo(img, CV_8U);
		  cv::applyColorMap(img, img, cv::COLORMAP_JET);
		  cv::imwrite("error3.png", img);

		return error;
	}else{
		  cv::Mat error = dispMap.clone();

		  for (int r = 0; r < dispMap.rows; r++) {
		    for (int c = 0; c < dispMap.cols; c++) {
		 		error.at<float>(r,c) = abs(dispMap.at<float>(r,c)-gt.at<float>(r,c));
			}
		    }
		  
		  
		  cv::Mat img;

		  error = error;

		  error.convertTo(img, CV_8U);
		  cv::applyColorMap(img, img, cv::COLORMAP_HOT);


		  for (int r = 0; r < dispMap.rows; r++) {
		    for (int c = 0; c < dispMap.cols; c++) {
				if(!isValid(gt.at<float>(r,c)) || !isValid(dispMap.at<float>(r,c))){
					img.at<cv::Vec<char, 3>>(r,c)[0] = 0;
		  			img.at<cv::Vec<char, 3>>(r,c)[1] = 0;
		  			img.at<cv::Vec<char, 3>>(r,c)[2] = 0;
				}
			  }
		  }


		  return img;

	}
}


float metric_avgerr(cv::Mat dmap, cv::Mat gt2, int maxdisp, bool clipDisp){
	
	int n = 0;
    int bad = 0;
    int invalid = 0;
    float serr = 0;
    for (int y = 0; y < gt2.rows; y++) {
		for (int x = 0; x < gt2.cols; x++) {
			float gt = gt2.at<float>(y,x);
			if (!isValid(gt)) // unknown
				continue;
			float d = dmap.at<float>(y,x);
			
			bool valid = isValid(d);
			if (valid) {
				if(clipDisp){
					float maxd = maxdisp; // max disp ranges
					// d = __max(0, __min(maxd, d));
					float val1 = maxd > d ? d:maxd;
					d = val1 > 0 ? val1 : 0;
				}
			}

			float err = fabs(d - gt);

			n++;
			if (valid) {
				serr += err;
			} else {// invalid (i.e. hole in sparse disp map)
				invalid++;
			}
		}
    }
	
    float avgErr = serr / (n - invalid); 
	
	return avgErr;
}


float metric_rmse(cv::Mat dmap, cv::Mat gt2, int maxdisp, bool clipDisp){
	int n = 0;
    int bad = 0;
    int invalid = 0;
    float serr = 0;
    for (int y = 0; y < gt2.rows; y++) {
		for (int x = 0; x < gt2.cols; x++) {
			float gt = gt2.at<float>(y,x);
			if (!isValid(gt)) // unknown
				continue;
			float d = dmap.at<float>(y,x);
			bool valid = isValid(d);
			if (valid) {
				if(clipDisp){
					float maxd = maxdisp; // max disp ranges
					// d = __max(0, __min(maxd, d));
					float val1 = maxd > d ? d:maxd;
					d = val1 > 0 ? val1 : 0;
				}
			}

			float err = fabs(d - gt);

			n++;
			if (valid) {
				serr += err*err;
			} else {// invalid (i.e. hole in sparse disp map)
				invalid++;
			}
			
		}
    }
    float badpercent =  100.0*bad/n;
    float invalidpercent =  100.0*invalid/n;
    float totalbadpercent =  100.0*(bad+invalid)/n;
    float avgErr = serr / (n - invalid);
	
	return std::sqrt(avgErr);
}

float metric_bad(cv::Mat dmap, cv::Mat gt2,float badthresh, int maxdisp, bool clipDisp){
	int n = 0;
    int bad = 0;
    int invalid = 0;
    float serr = 0;
    for (int y = 0; y < gt2.rows; y++) {
		for (int x = 0; x < gt2.cols; x++) {
			float gt = gt2.at<float>(y,x);
			if (!isValid(gt)) // unknown
				continue;
			float d = dmap.at<float>(y,x);
			bool valid = isValid(d);
			if (valid) {
				if(clipDisp){
					float maxd = maxdisp; // max disp ranges
					// d = __max(0, __min(maxd, d));
					float val1 = maxd > d ? d:maxd;
					d = val1 > 0 ? val1 : 0;
				}
			}

			float err = fabs(d - gt);

			n++;
			if (valid) {
				serr += err;
				if (err > badthresh) {
					bad++;
				}
			} else {// invalid (i.e. hole in sparse disp map)
				invalid++;
			}
			
		}
    }
    float badpercent =  100.0*bad/n;
    float invalidpercent =  invalid/n;
    float totalbadpercent =  (bad+invalid)/n;
    float avgErr = serr / (n - invalid);
	
	return badpercent;
}

float metric_D1ALL(cv::Mat dmap, cv::Mat gt){
  float total = 0;
  float invalid = 0;
  for(int r =0;r< dmap.rows;r++){
    for(int c = 0;c < dmap.cols;c++){
      if(isValid(dmap.at<float>(r,c)) && isValid(gt.at<float>(r,c))){
        total++;
        float E = fabs(dmap.at<float>(r,c) - gt.at<float>(r,c));
        if((E > 3 && (E/gt.at<float>(r,c)) > 0.05))
          invalid++;
        }
      }
  }
  return total != 0 ? 100.0*(invalid/total) : -1;
}



float metric_density(cv::Mat dmap, cv::Mat gt){
  float total = 0;
  float valid = 0;
  for(int r =0;r< dmap.rows;r++){
    for(int c = 0;c < dmap.cols;c++){
        if(isValid(dmap.at<float>(r,c))){
          valid++;
        }
        total++;
    }
  }
  return total != 0 ? 100.0*(valid/total) : -1;
}







void run(bool sparse, std::string methodName,std::string versionString,std::string resultsFile_String, std::string leftImg_string,std::string rightImg_string,std::string dispImg_string,std::string dispImgVis_string,int disparityLevels,int fator,float alpha,int minAreaMatchedFineTopNode,int nColors,int * sizes,int total,uint kernelCostComp,int kernelCostVolume,int everyNthRow, float minConfidencePercentage,float allowanceSearchRange){

	cv::Mat img_l = cv::imread(leftImg_string.c_str());
	cv::Mat img_r = cv::imread(rightImg_string.c_str());

	//printf("fator: %d\n",fator);

	int maxAreaMatchedFineTopNode = img_l.cols / fator;

	if (img_l.cols > 0) {
		img_l = preProcess(img_l);
		img_r = preProcess(img_r);

		double startTotal = getWallTime();

		cv::Mat img = work(img_l, img_r, sparse, alpha, minAreaMatchedFineTopNode, maxAreaMatchedFineTopNode, nColors, sizes, total, kernelCostComp,kernelCostVolume, everyNthRow, disparityLevels, minConfidencePercentage,allowanceSearchRange);

		writeInfo(methodName, versionString, startTotal, img, resultsFile_String,leftImg_string);
		writeDispmap(img, dispImg_string);
		writeVisDispmap(img, dispImgVis_string, disparityLevels);
		img.release();
	};
};



cv::Mat readbinfile(std::string gt_string,int width,int height){
    std::ifstream inputfile(gt_string.c_str(),std::ifstream::binary);

	cv::Mat gt = cv::Mat(height,width,CV_32F);;

	for (int x = 0; x < width; x++) {
	  for (int y = 0; y < height; y++) {

			unsigned char temp[sizeof(float)];
			inputfile.read(reinterpret_cast<char*>(temp), sizeof(float));
			unsigned char t = temp[0];
			temp[0] = temp[3];
			temp[3] = t;
			t = temp[1];
			temp[1] = temp[2];
			temp[2] = t;

			gt.at<float>(y,x) = (float)reinterpret_cast<float&>(temp);
		}
	}

	return gt;
}


bool checkifalldispmapshavebeenconsidered = false;
bool checkdensitygt100 = false;
int methodtorun = 0;// 0 = mtstereo, 1 = sgbm1, 2 = sgbm2, 3 = sed, 4 = elas


void calcMetricsMasked(std::string methodName,std::string versionString,std::string resultsFile_String, std::string dispImg_string,std::string gt_string,int dispLevels,std::string mask_string, bool flip){
  cv::Mat dmap;
  cv::Mat gt;
  if(ends_with(dispImg_string.c_str(), ".pfm"))
	 dmap = opencv_pfm::imread_pfm(dispImg_string.c_str());
  if(ends_with(dispImg_string.c_str(), ".png"))
   dmap = cv::imread(dispImg_string.c_str());
  if(ends_with(gt_string.c_str(), ".pfm"))
	 gt = opencv_pfm::imread_pfm(gt_string.c_str());
  if(ends_with(gt_string.c_str(), ".png"))
   gt = cv::imread(gt_string.c_str());
  if(ends_with(gt_string.c_str(), ".bin"))
   gt = readbinfile(gt_string,dmap.cols,dmap.rows);




	if(checkifalldispmapshavebeenconsidered){
		
		if(dmap.cols > 0 && gt.cols > 0){
			printf("%s\n",gt_string.c_str());
		}
			
	}else{
	

	if (dmap.cols > 0 && gt.cols > 0){
		
		if (dmap.channels() > 1){
			cv::cvtColor(dmap, dmap, cv::COLOR_RGB2GRAY);
		}
		if (gt.channels() > 1){
			cv::cvtColor(gt, gt, cv::COLOR_RGB2GRAY);
		}	
		
		dmap.convertTo(dmap,CV_32FC1);
		gt.convertTo(gt,CV_32FC1);
		
		
		if(flip)
			cv::flip(dmap,dmap,0);


		// kitti2015 invalid
		if(strcmp(methodName.c_str(), "kitti2015") == 0){
			 for(int r =0;r<gt.rows;r++){
				 for(int c =0;c<gt.cols;c++){
					float val = gt.at<float>(r,c);
					if(val == 0){
						gt.at<float>(r,c) = -1;
					}
				 }
			 }

			for(int r =0;r<dmap.rows;r++){
				for(int c =0;c<dmap.cols;c++){
					if(dmap.at<float>(r,c) <= 0 || dmap.at<float>(r,c) > 100000){
						dmap.at<float>(r,c) = -1;
					}
				 }
			 }


		}
	
		/*
	
		if(checkdensitygt100){
			  float total = 0;
			  float valid = 0;
			  for(int r =0;r< gt.rows;r++){
				for(int c = 0;c < gt.cols;c++){
				  if(isValid(gt.at<float>(r,c))){
					valid++;
				  }
				  total++;
				}
			  }
			  if(valid != total){
				  printf("ERROR! - Density GT:%f \n",total != 0 ? 100.0*(valid/total) : -1);
			  }else{
				  printf("ok\n");
			  }
			  return;
		}
		
		float minval = 100000000;
		float maxval = 0;
		for(int r =0;r<gt.rows;r++){
				 for(int c =0;c<gt.cols;c++){					 
					float val = gt.at<float>(r,c);
					if(isValid(val)){
						if(val > maxval)
							maxval = val;
						if(val < minval)
							minval = val;	
					}
			 }
		}
		//printf("minval in gt: %f max val in gt: %f\n",minval,maxval);
		*/

		

		/*
		// our maps invalid
		for(int r =0;r<dmap.rows;r++){
			for(int c =0;c<dmap.cols;c++){
				if(dmap.at<float>(r,c) <= 0 || dmap.at<float>(r,c) > 100000){
					dmap.at<float>(r,c) = -1;
				}
			 }
		 }
		*/
		
		float avgerr = metric_avgerr(dmap, gt, dispLevels,false);
		float rmse = 0;//metric_rmse(dmap, gt, dispLevels,true);
		float bad1 = 0;//metric_bad(dmap, gt, 1, dispLevels,true);
		float bad2 = 0;//metric_bad(dmap, gt, 2, dispLevels,true);
		float bad3 = 0;//metric_bad(dmap, gt, 3, dispLevels,true);
		float bad4 = 0;//metric_bad(dmap, gt, 4, dispLevels,true);
		float bad5 = 0;//metric_bad(dmap, gt, 5, dispLevels,true);	
		
		
		float avgerr_clipped = 0;//metric_avgerr(dmap, gt, dispLevels,false);
		float rmse_clipped = 0;//metric_rmse(dmap, gt, dispLevels,false);
		float bad1_clipped = 0; //metric_bad(dmap, gt, 1, dispLevels,false);
		float bad2_clipped = 0;//metric_bad(dmap, gt, 2, dispLevels,false);
		float bad3_clipped = 0;//metric_bad(dmap, gt, 3, dispLevels,false);
		float bad4_clipped = 0;//metric_bad(dmap, gt, 4, dispLevels,false);
		float bad5_clipped = 0;//metric_bad(dmap, gt, 5, dispLevels,false);	
		
		float D1ALL = 0;//metric_D1ALL(dmap, gt); 	
		float density = metric_density(dmap, gt);     
				
		writeInfoResults(resultsFile_String, methodName, versionString, resultsFile_String, dispImg_string, avgerr, rmse, bad1, bad2, bad3, bad4, bad5,D1ALL,density,avgerr_clipped, rmse_clipped, bad1_clipped, bad2_clipped, bad3_clipped, bad4_clipped, bad5_clipped);
		
		if(false){
			cv::Mat error = writeVisError(gt, dmap,dispLevels);
			//cv::imwrite(dispImg_string+"_error.png",error);
		}
		}
	}
}

void calcMetrics(std::string methodName,std::string versionString,std::string resultsFile_String, std::string dispImg_string,std::string gt_string,int dispLevels,bool flip){
	calcMetricsMasked( methodName, versionString, resultsFile_String,  dispImg_string, gt_string, dispLevels, "notmasked",flip);
}

std::string genVersionString(int versionIndex, bool * useOldGradCost, bool * sparse, bool * useBilateral){
    std::string versionString0;
    std::string versionString1;
    std::string versionString2;


	*useOldGradCost = false;
	*useBilateral = true;
	versionString0 = "ng";
	versionString1 = "_bi";
    if (versionIndex == 0) {
		*sparse = true;
		versionString2 = "_sp";
    }

    if (versionIndex == 1) {
		*sparse = false;
		versionString2 = "_sd";
    }

    //return versionString0+versionString1+versionString2+"v1";
    return versionString0+versionString1+versionString2;
    //return "mts1"+versionString2;
}


bool flyingthingsExclude(std::string filename){
	std::string endsWithExclude[244] = {"A/0005/left/0006.png ","A/0005/left/0007.png ","A/0005/left/0008.png ","A/0005/left/0009.png ","A/0005/left/0010.png ","A/0005/left/0011.png ","A/0005/left/0012.png ","A/0005/left/0013.png ","A/0005/left/0014.png ","A/0005/left/0015.png ","A/0005/right/0006.png ","A/0005/right/0007.png ","A/0005/right/0008.png ","A/0005/right/0009.png ","A/0005/right/0010.png ","A/0005/right/0011.png ","A/0005/right/0012.png ","A/0005/right/0013.png ","A/0005/right/0014.png ","A/0005/right/0015.png ","A/0008/left/0006.png ","A/0008/left/0007.png ","A/0008/left/0008.png ","A/0008/left/0009.png ","A/0008/left/0010.png ","A/0008/left/0011.png ","A/0008/left/0012.png ","A/0008/left/0013.png ","A/0008/left/0014.png ","A/0008/left/0015.png ","A/0008/right/0006.png ","A/0008/right/0007.png ","A/0008/right/0008.png ","A/0008/right/0009.png ","A/0008/right/0010.png ","A/0008/right/0011.png ","A/0008/right/0012.png ","A/0008/right/0013.png ","A/0008/right/0014.png ","A/0008/right/0015.png ","A/0013/left/0015.png ","A/0013/right/0015.png ","A/0031/left/0006.png ","A/0031/left/0007.png ","A/0031/left/0008.png ","A/0031/left/0009.png ","A/0031/left/0010.png ","A/0031/left/0011.png ","A/0031/left/0012.png ","A/0031/left/0013.png ","A/0031/left/0014.png ","A/0031/left/0015.png ","A/0031/right/0006.png ","A/0031/right/0007.png ","A/0031/right/0008.png ","A/0031/right/0009.png ","A/0031/right/0010.png ","A/0031/right/0011.png ","A/0031/right/0012.png ","A/0031/right/0013.png ","A/0031/right/0014.png ","A/0031/right/0015.png ","A/0110/left/0006.png ","A/0110/left/0007.png ","A/0110/left/0008.png ","A/0110/left/0009.png ","A/0110/left/0010.png ","A/0110/left/0011.png ","A/0110/left/0012.png ","A/0110/left/0013.png ","A/0110/left/0014.png ","A/0110/left/0015.png ","A/0110/right/0006.png ","A/0110/right/0007.png ","A/0110/right/0008.png ","A/0110/right/0009.png ","A/0110/right/0010.png ","A/0110/right/0011.png ","A/0110/right/0012.png ","A/0110/right/0013.png ","A/0110/right/0014.png ","A/0110/right/0015.png ","A/0123/left/0006.png ","A/0123/left/0007.png ","A/0123/left/0008.png ","A/0123/left/0009.png ","A/0123/left/0010.png ","A/0123/left/0011.png ","A/0123/left/0012.png ","A/0123/left/0013.png ","A/0123/left/0014.png ","A/0123/left/0015.png ","A/0123/right/0006.png ","A/0123/right/0007.png ","A/0123/right/0008.png ","A/0123/right/0009.png ","A/0123/right/0010.png ","A/0123/right/0011.png ","A/0123/right/0012.png ","A/0123/right/0013.png ","A/0123/right/0014.png ","A/0123/right/0015.png ","A/0149/left/0006.png ","A/0149/left/0007.png ","A/0149/left/0008.png ","A/0149/left/0009.png ","A/0149/left/0010.png ","A/0149/left/0011.png ","A/0149/left/0012.png ","A/0149/left/0013.png ","A/0149/left/0014.png ","A/0149/left/0015.png ","A/0149/right/0006.png ","A/0149/right/0007.png ","A/0149/right/0008.png ","A/0149/right/0009.png ","A/0149/right/0010.png ","A/0149/right/0011.png ","A/0149/right/0012.png ","A/0149/right/0013.png ","A/0149/right/0014.png ","A/0149/right/0015.png ","B/0046/left/0006.png ","B/0046/left/0007.png ","B/0046/left/0008.png ","B/0046/left/0009.png ","B/0046/left/0010.png ","B/0046/left/0011.png ","B/0046/left/0012.png ","B/0046/left/0013.png ","B/0046/left/0014.png ","B/0046/left/0015.png ","B/0046/right/0006.png ","B/0046/right/0007.png ","B/0046/right/0008.png ","B/0046/right/0009.png ","B/0046/right/0010.png ","B/0046/right/0011.png ","B/0046/right/0012.png ","B/0046/right/0013.png ","B/0046/right/0014.png ","B/0046/right/0015.png ","B/0048/left/0006.png ","B/0048/left/0007.png ","B/0048/left/0008.png ","B/0048/left/0009.png ","B/0048/left/0010.png ","B/0048/left/0011.png ","B/0048/left/0012.png ","B/0048/left/0013.png ","B/0048/left/0014.png ","B/0048/left/0015.png ","B/0048/right/0006.png ","B/0048/right/0007.png ","B/0048/right/0008.png ","B/0048/right/0009.png ","B/0048/right/0010.png ","B/0048/right/0011.png ","B/0048/right/0012.png ","B/0048/right/0013.png ","B/0048/right/0014.png ","B/0048/right/0015.png ","B/0074/left/0006.png ","B/0074/left/0007.png ","B/0074/left/0008.png ","B/0074/left/0009.png ","B/0074/left/0010.png ","B/0074/left/0011.png ","B/0074/left/0012.png ","B/0074/left/0013.png ","B/0074/left/0014.png ","B/0074/left/0015.png ","B/0074/right/0006.png ","B/0074/right/0007.png ","B/0074/right/0008.png ","B/0074/right/0009.png ","B/0074/right/0010.png ","B/0074/right/0011.png ","B/0074/right/0012.png ","B/0074/right/0013.png ","B/0074/right/0014.png ","B/0074/right/0015.png ","B/0078/left/0006.png ","B/0078/left/0007.png ","B/0078/left/0008.png ","B/0078/left/0009.png ","B/0078/left/0010.png ","B/0078/left/0011.png ","B/0078/left/0012.png ","B/0078/left/0013.png ","B/0078/left/0014.png ","B/0078/left/0015.png ","B/0078/right/0006.png ","B/0078/right/0007.png ","B/0078/right/0008.png ","B/0078/right/0009.png ","B/0078/right/0010.png ","B/0078/right/0011.png ","B/0078/right/0012.png ","B/0078/right/0013.png ","B/0078/right/0014.png ","B/0078/right/0015.png ","B/0133/left/0006.png ","B/0133/right/0006.png ","B/0136/left/0006.png ","B/0136/left/0007.png ","B/0136/left/0008.png ","B/0136/left/0009.png ","B/0136/left/0010.png ","B/0136/left/0011.png ","B/0136/left/0012.png ","B/0136/left/0013.png ","B/0136/left/0014.png ","B/0136/left/0015.png ","B/0136/right/0006.png ","B/0136/right/0007.png ","B/0136/right/0008.png ","B/0136/right/0009.png ","B/0136/right/0010.png ","B/0136/right/0011.png ","B/0136/right/0012.png ","B/0136/right/0013.png ","B/0136/right/0014.png ","B/0136/right/0015.png ","B/0138/left/0006.png ","B/0138/left/0007.png ","B/0138/left/0008.png ","B/0138/left/0009.png ","B/0138/left/0010.png ","B/0138/left/0011.png ","B/0138/left/0012.png ","B/0138/left/0013.png ","B/0138/left/0014.png ","B/0138/left/0015.png ","B/0138/right/0006.png ","B/0138/right/0007.png ","B/0138/right/0008.png ","B/0138/right/0009.png ","B/0138/right/0010.png ","B/0138/right/0011.png ","B/0138/right/0012.png ","B/0138/right/0013.png ","B/0138/right/0014.png ","B/0138/right/0015.png"};
	
	for(int i =0;i<244;i++){
		if(ends_with(filename, endsWithExclude[i]))
			return true;
	}
	return false;
}


bool dbCheckMode = false;

void processMethod(const char * methodName) {

 float minConfidencePercentage = 0;
  float allowanceSearchRange = 0;

  bool * sparse = (bool *) malloc(sizeof(bool));
  std::string resultsFile_String;
  std::string resultsFileAccuracy_String;
  std::string versionString;
  bool * useOldGradCost = (bool *) malloc(sizeof(bool));
  bool * useBilateral = (bool *)  malloc(sizeof(bool));

  if(isResultsCompMode){
      if (strcmp(methodName, "middleburry") == 0) {
        resultsFile_String = rootFolder+"/middleburry/results.csv";
        initializeResultsFile(resultsFile_String);
      } else if (strcmp(methodName, "realgarden") == 0) {
        resultsFile_String = rootFolder+"/new/Trimbot2020GardenNew/results.csv";
        initializeResultsFile(resultsFile_String);
      } else if (strcmp(methodName, "synthgarden") == 0) {
        resultsFile_String = rootFolder+"/new/SyntheticGardenNew/results.csv";
        initializeResultsFile(resultsFile_String);
      } else if (strcmp(methodName, "kitti2015") == 0) {
        resultsFile_String = rootFolder+"/kitti2015/results.csv";
        initializeResultsFile(resultsFile_String);
      } else if (strcmp(methodName, "driving") == 0) {
        resultsFile_String = rootFolder+"/driving/results.csv";
        initializeResultsFile(resultsFile_String);
      } else if (strcmp(methodName, "monkaa") == 0) {
        resultsFile_String = rootFolder+"/monkaa/results.csv";
        initializeResultsFile(resultsFile_String);
      } else if (strcmp(methodName, "flyingthings") == 0) {
        resultsFile_String = rootFolder+"/flyingthings3D/results.csv";
        initializeResultsFile(resultsFile_String);
      }
  }
  if(isMetricCompMode){
      if (strcmp(methodName, "middleburry") == 0) {
        resultsFileAccuracy_String = rootFolder+"/middleburry/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      } else if (strcmp(methodName, "realgarden") == 0) {
        resultsFileAccuracy_String = rootFolder+"/new/Trimbot2020GardenNew/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      } else if (strcmp(methodName, "synthgarden") == 0) {
        resultsFileAccuracy_String = rootFolder+"/new/SyntheticGardenNew/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      } else if (strcmp(methodName, "kitti2015") == 0) {
        resultsFileAccuracy_String = rootFolder+"/kitti2015/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      } else if (strcmp(methodName, "driving") == 0) {
        resultsFileAccuracy_String = rootFolder+"/driving/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      } else if (strcmp(methodName, "monkaa") == 0) {
        resultsFileAccuracy_String = rootFolder+"/monkaa/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      } else if (strcmp(methodName, "flyingthings") == 0) {
        resultsFileAccuracy_String = rootFolder+"/flyingthings3D/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      }
  }

  if (strcmp(methodName, "middleburry") == 0) {
    std::string folders[15] = {"Adirondack", "ArtL", "Jadeplant", "Motorcycle", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "Playtable", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"}; 
    int disps[15] = {290, 256, 640, 280, 280, 260, 260, 300, 330, 290, 290, 260, 240, 256, 760 }; 

    for (int i = 0; i < 15; i++) {
      std::string folder = folders[i];
      for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
		
        versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
        int dispLevels = disps[i];

        if(isResultsCompMode){
          std::string leftImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/im0.png";
          std::string rightImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/im1.png";
          std::string dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0Mt_" + versionString + ".pfm";
          std::string dispImgVis_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0Mt_" + versionString + ".png";
          int sizes[5] = {1,0,0,0};
		
		  int nColors = (*sparse)?16:8;
		  
		   
		  if(dbCheckMode){
			  if(versionIndex !=0) 
				  continue;
			  printf("%s\n",leftImg_string.c_str());
		  }else{
			  		  
	  if(methodtorun == 0){			  		
			  run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,2,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
		}else{
			if(versionIndex == 0){
				if(methodtorun == 1){
					versionString = "sgbm1";

					dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".pfm";
		 			dispImgVis_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".png";

					printf("python sgbm1.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
				}
				if(methodtorun == 2){
					versionString = "sgbm2";

					dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".pfm";
		 			dispImgVis_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".png";

					printf("python sgbm2.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
				}
				if(methodtorun == 3){
					versionString = "sed";

					dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".pfm";

					printf("./SED/experiments/sed_run/sed_run %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
				}
				if(methodtorun == 4){
					versionString = "elas";

					dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".pfm";
					
					//std::string rightImg_string_ = rightImg_string+"";
					//std::string leftImg_string_ = leftImg_string+"";

		
					//leftImg_string_ = leftImg_string_.substr(0, leftImg_string_.size()-4)+".pgm";
					//rightImg_string_ = rightImg_string_.substr(0, rightImg_string_.size()-4)+".pgm";	

					//printf("convert %s -flatten %s \n",leftImg_string.c_str(),leftImg_string_.c_str());
					//printf("convert %s -flatten %s \n",rightImg_string.c_str(),rightImg_string_.c_str());

					printf("./elas/elas %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
				}
			}
		}
	}




        }
        if(isMetricCompMode){
          std::string dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0Mt_" + versionString + ".pfm";
          std::string gt_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0GT.pfm";
		  //std::string mask_string = rootFolder+"/middleburry/trainingF/" + folder + "/mask0nocc.png";
		  
		  
		  if(versionIndex == 0){
				if(methodtorun == 1){
					versionString = "sgbm1";
					dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".pfm";
				}
				if(methodtorun == 2){
					versionString = "sgbm2";
					dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".pfm";
				}
				if(methodtorun == 3){
					versionString = "sed";
					dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".pfm";

				}
				if(methodtorun == 4){
					versionString = "elas";
					dispImg_string = rootFolder+"/middleburry/trainingF/" + folder + "/disp0_" + versionString + ".pfm";
				}
			}
		  
		  
		  if(methodtorun == 0 || (methodtorun > 0 && versionIndex ==0))
			calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string, dispLevels,false);
        }
      }
	  
    }
  } else if (strcmp(methodName, "realgarden") == 0) {
        for (int k = 1; k <= 1270; k += 1) {

            for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
              versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
                int dispLevels = 35;
                if(isResultsCompMode){
                  std::string leftImg_string = rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + ".jpg";
                  std::string rightImg_string = rootFolder+"/new/Trimbot2020GardenNew/TEST/right/"+ std::to_string(k) + ".jpg";
                  std::string dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
                  std::string dispImgVis_string = rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_dispVis_" + versionString + ".png";
                  int sizes[5] = {1,0,0,0};
				  
				  int nColors = (*sparse)?16:8;
				  		  if(dbCheckMode){
			  if(versionIndex !=0) 
				  continue;
			  printf("%s\n",leftImg_string.c_str());
		  }else{
			  
			  
			  
			  
			  
			  
		if(methodtorun == 0){		
               run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,2,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
	
		}else{
			if(versionIndex == 0){
				if(methodtorun == 1){
					versionString = "sgbm1";

					dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
					dispImgVis_string = rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_dispVis_" + versionString + ".png";

					printf("python sgbm1.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
				}
				if(methodtorun == 2){
					versionString = "sgbm2";

					dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
					dispImgVis_string = rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_dispVis_" + versionString + ".png";

					printf("python sgbm2.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
				}
				if(methodtorun == 3){
					versionString = "sed";

					dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".png";

					printf("./SED/experiments/sed_run/sed_run %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
				}
				if(methodtorun == 4){
					versionString = "elas";

					dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";

					printf("./elas/elas %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
				}
			}
		}
			  
			

		}
				}
                if(isMetricCompMode){
				  std::string dispImg_string = rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
				  std::string gt_string = rootFolder+"/new/Trimbot2020GardenNew/TEST/ground_truth_disparity/"+ std::to_string(k) + ".png";
				  
					if(versionIndex == 0){
						if(methodtorun == 1){
							versionString = "sgbm1";

							dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
						}
						if(methodtorun == 2){
							versionString = "sgbm2";

							dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
						}
						if(methodtorun == 3){
							versionString = "sed";

							dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".png";
						}
						if(methodtorun == 4){
							versionString = "elas";

							dispImg_string =  rootFolder+"/new/Trimbot2020GardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
						}
					}
				  
				  
			if(methodtorun == 0 || (methodtorun > 0 && versionIndex ==0))
				  calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string,dispLevels,true);
                }
            }
			
        }
  } else if (strcmp(methodName, "synthgarden") == 0) {

        for (int k = 4601; k <= 5021; k += 1) {
            for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
              versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
              int dispLevels = 100;
                if(isResultsCompMode){
                 std::string leftImg_string = rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + ".jpg";
                  std::string rightImg_string = rootFolder+"/new/SyntheticGardenNew/TEST/right/"+ std::to_string(k) + ".jpg";
                  std::string dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
                  std::string dispImgVis_string = rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_dispVis_" + versionString + ".png";
                  int sizes[5] = {1,0,0,0};
				  
				  int nColors = (*sparse)?16:8;

		  if(dbCheckMode){
			  if(versionIndex !=0) 
				  continue;
			  printf("%s\n",leftImg_string.c_str());
		  }else{
			  
			  
			  
			  
			  
			  
		if(methodtorun == 0){		
                  run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,2,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
		}else{
			if(versionIndex == 0){
				if(methodtorun == 1){
					versionString = "sgbm1";

					dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
					dispImgVis_string = rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_dispVis_" + versionString + ".png";

					printf("python sgbm1.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
				}
				if(methodtorun == 2){
					versionString = "sgbm2";

					dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
					dispImgVis_string = rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_dispVis_" + versionString + ".png";

					printf("python sgbm2.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
				}
				if(methodtorun == 3){
					versionString = "sed";

					dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".png";

					printf("./SED/experiments/sed_run/sed_run %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
				}
				if(methodtorun == 4){
					versionString = "elas";

					dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";

					printf("./elas/elas %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
				}
			}
		}
			  
			  
                 
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  }
				}
                if(isMetricCompMode){
				  std::string dispImg_string = rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
				  std::string gt_string = rootFolder+"/new/SyntheticGardenNew/TEST/ground_truth_disparity/"+ std::to_string(k) + ".png";
				  
				  
				  
				  	if(versionIndex == 0){
				if(methodtorun == 1){
					versionString = "sgbm1";

					dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
				}
				if(methodtorun == 2){
					versionString = "sgbm2";

					dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
				}
				if(methodtorun == 3){
					versionString = "sed";

					dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".png";
				}
				if(methodtorun == 4){
					versionString = "elas";

					dispImg_string =  rootFolder+"/new/SyntheticGardenNew/TEST/left/"+ std::to_string(k) + "_disp_" + versionString + ".pfm";
				}
			}
				  
				  
				  
		 if(methodtorun == 0 || (methodtorun > 0 && versionIndex ==0))
                 	 calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string,dispLevels,true);
                }
            }
			
        }
    
  } else if (strcmp(methodName, "kitti2015") == 0) {
    
    for (int k = 0; k <= 199; k += 1) {
      for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
        versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
        int dispLevels = 255;
        if(isResultsCompMode){
	  
	  

          std::string leftImg_string = rootFolder+"/kitti2015/training/image_2/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          std::string rightImg_string = rootFolder+"/kitti2015/training/image_3/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          std::string dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          std::string dispImgVis_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/visualization_" + zeroPadded(std::to_string(k), 6) + "_10.png";
          int sizes[5] = {1,0,0,0};
		  
		  int nColors = (*sparse)?16:8;
		  
		  if(dbCheckMode){
			  if(versionIndex !=0) 
				  continue;
			  printf("%s\n",leftImg_string.c_str());
		  }else{
			  
			  
		if(methodtorun == 0){		
			 mkdir2(rootFolder+"/kitti2015/training/disp_" + versionString);
			 run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,2,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
		
		}else{
			if(versionIndex == 0){
				if(methodtorun == 1){
					versionString = "sgbm1";

					dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.pfm";
					dispImgVis_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/visualization_" + zeroPadded(std::to_string(k), 6) + "_10.png";

					printf("python sgbm1.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
				}
				if(methodtorun == 2){
					versionString = "sgbm2";

					dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.pfm";
					dispImgVis_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/visualization_" + zeroPadded(std::to_string(k), 6) + "_10.png";

					printf("python sgbm2.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
				}
				if(methodtorun == 3){
					versionString = "sed";

					dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.png";

					printf("./SED/experiments/sed_run/sed_run %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
				}
				if(methodtorun == 4){
					versionString = "elas";

					dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.pfm";

					printf("./elas/elas %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
				}
				mkdir2(rootFolder+"/kitti2015/training/disp_" + versionString);
			}
		}
			  
			  
         







			}
        }
        if(isMetricCompMode){
          std::string dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          std::string gt_string = rootFolder+"/kitti2015/training/disp_occ_0/" + zeroPadded(std::to_string(k), 6) + "_10.png";
		  
		  
		  
			if(versionIndex == 0){
				if(methodtorun == 1){
					versionString = "sgbm1";

					dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.pfm";
				}
				if(methodtorun == 2){
					versionString = "sgbm2";

					dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.pfm";
				}
				if(methodtorun == 3){
					versionString = "sed";

					dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.png";
				}
				if(methodtorun == 4){
					versionString = "elas";

					dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.pfm";
				}
			}
		  
		  
		  
		if(methodtorun == 0 || (methodtorun > 0 && versionIndex ==0))
          calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string,dispLevels,false);
        }
      }  
	  	  
    }
  } else if (strcmp(methodName, "driving") == 0) {
 
	for (int forwardsOrBackwards = 0; forwardsOrBackwards <= 1; forwardsOrBackwards++) {		
		for (int i = 0; i <= 300; i += 1) {			
		  for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
			versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
			int dispLevels = 255;
			
			std::string forwardbackwardString = "scene_forwards";
			if(forwardsOrBackwards == 1)
				forwardbackwardString = "scene_backwards";
			
			if(isResultsCompMode){
			  std::string leftImg_string = rootFolder+"/driving/frames_cleanpass/35mm_focallength/"+ forwardbackwardString+"/fast/left/" + zeroPadded(std::to_string(i), 4) + ".png";
			  std::string rightImg_string = rootFolder+"/driving/frames_cleanpass/35mm_focallength/"+ forwardbackwardString+"/fast/right/" + zeroPadded(std::to_string(i), 4) + ".png";
			  std::string dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";
			  std::string dispImgVis_string = rootFolder+"/driving/disparity_" + versionString +  "/dispvis_" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".png";
			  int sizes[5] = {1,0,0,0};
			  
			  int nColors = (*sparse)?16:8;
			  
			  		  if(dbCheckMode){
			  if(versionIndex !=0) 
				  continue;
			  printf("%s\n",leftImg_string.c_str());
		  }else{
			  
			  
			  
			  
			  if(methodtorun == 0){	
				  mkdir2(rootFolder+"/driving/disparity_" + versionString);
				  run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,2,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
			}else{
				if(versionIndex == 0){
					if(methodtorun == 1){
						versionString = "sgbm1";

						dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";
						dispImgVis_string = rootFolder+"/driving/disparity_" + versionString +  "/dispvis_" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".png";

						printf("python sgbm1.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
					}
					if(methodtorun == 2){
						versionString = "sgbm2";

						dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";
						dispImgVis_string = rootFolder+"/driving/disparity_" + versionString +  "/dispvis_" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".png";

						printf("python sgbm2.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
					}
					if(methodtorun == 3){
						versionString = "sed";

						dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";

						printf("./SED/experiments/sed_run/sed_run %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
					}
					if(methodtorun == 4){
						versionString = "elas";

						dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";

						printf("./elas/elas %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
					}
				}
				mkdir2(rootFolder+"/driving/disparity_" + versionString);
			}
			  
			  
			  
			  
			  
			
		  }
			}
			if(isMetricCompMode){
			  std::string dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";
			  std::string gt_string = rootFolder+"/driving/disparity/35mm_focallength/"+forwardbackwardString+"/fast/left/" + zeroPadded(std::to_string(i), 4) + ".pfm";
			  
			  
			  
			  
				if(versionIndex == 0){
					if(methodtorun == 1){
						versionString = "sgbm1";

						dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";
					}
					if(methodtorun == 2){
						versionString = "sgbm2";

						dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";
					}
					if(methodtorun == 3){
						versionString = "sed";

						dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";
					}
					if(methodtorun == 4){
						versionString = "elas";

						dispImg_string = rootFolder+"/driving/disparity_" + versionString + "/" + forwardbackwardString + zeroPadded(std::to_string(i), 4) + ".pfm";
					}
				}
			  
			  
			  
			 if(methodtorun == 0 || (methodtorun > 0 && versionIndex ==0))
				calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string,dispLevels,false);
			}
		  }  
		  	  
		}
	}
  } else if (strcmp(methodName, "monkaa") == 0) {
    mkdir2(rootFolder+"/monkaa/disparity_" + versionString);

    std::string folders[30] = {"a_rain_of_stones_x2", "funnyworld_camera2_x2", "eating_camera2_x2", "funnyworld_x2", "eating_naked_camera2_x2", "lonetree_augmented0_x2", "eating_x2", "lonetree_augmented1_x2", "family_x2", "lonetree_difftex2_x2", "flower_storm_augmented0_x2", "lonetree_difftex_x2", "flower_storm_augmented1_x2", "lonetree_winter_x2", "flower_storm_x2", "lonetree_x2", "funnyworld_augmented0_x2", "top_view_x2", "funnyworld_augmented1_x2", "treeflight_augmented0_x2", "funnyworld_camera2_augmented0_x2", "treeflight_augmented1_x2", "funnyworld_camera2_augmented1_x2", "treeflight_x2"};

    for (int j = 0; j < 24; j += 1) {
      for (int i = 0; i <= 500; i += 1) {
        std::string folder = folders[j];

        for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
          versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
          int dispLevels = 255;

          if(isResultsCompMode){
            std::string leftImg_string = rootFolder+"/monkaa/frames_cleanpass/" + folder + "/left/" + zeroPadded(std::to_string(i), 4) + ".png";
            std::string rightImg_string = rootFolder+"/monkaa/frames_cleanpass/" + folder + "/right/" + zeroPadded(std::to_string(i), 4) + ".png";
            std::string dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/" +folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";
            std::string dispImgVis_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+ folder + "/dispVis_" + zeroPadded(std::to_string(i), 4) + ".png";
            
            int sizes[5] = {1,0,0,0};
			
		      	int nColors = (*sparse)?16:8;
		   if(dbCheckMode){
			  if(versionIndex !=0) 
				  continue;
			  printf("%s\n",leftImg_string.c_str());
		  }else{


			  if(methodtorun == 0){	
				  mkdir2(rootFolder+"/monkaa/disparity_" + versionString + "/" + folder);
				  run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,2,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
			}else{
				if(versionIndex == 0){
					if(methodtorun == 1){
						versionString = "sgbm1";

						dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";
						dispImgVis_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+ folder + "/dispVis_" + zeroPadded(std::to_string(i), 4) + ".png";

						printf("python sgbm1.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
					}
					if(methodtorun == 2){
						versionString = "sgbm2";

						dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";
						dispImgVis_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+ folder + "/dispVis_" + zeroPadded(std::to_string(i), 4) + ".png";

						printf("python sgbm2.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
					}
					if(methodtorun == 3){
						versionString = "sed";

						dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";

						printf("./SED/experiments/sed_run/sed_run %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
					}
					if(methodtorun == 4){
						versionString = "elas";

						dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";

						std::string rightImg_string_ = rightImg_string+"";
						std::string leftImg_string_ = leftImg_string+"";
	
			
						leftImg_string_ = leftImg_string_.substr(0, leftImg_string_.size()-4)+".pgm";
						rightImg_string_ = rightImg_string_.substr(0, rightImg_string_.size()-4)+".pgm";	

						//printf("convert %s -flatten %s \n",leftImg_string.c_str(),leftImg_string_.c_str());
						//printf("convert %s -flatten %s \n",rightImg_string.c_str(),rightImg_string_.c_str());

						printf("./elas/elas %s %s %s %d\n",leftImg_string_.c_str(),rightImg_string_.c_str(),dispImg_string.c_str(),dispLevels);

					}
				}
				mkdir2(rootFolder+"/monkaa/disparity_" + versionString + "/" + folder);
			}




            		
		  }
          }
          if(isMetricCompMode){
            std::string dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/" +folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";
            std::string gt_string = rootFolder+"/monkaa/disparity/" +folder+"/left/"+ zeroPadded(std::to_string(i), 4) + ".pfm";
			


		if(versionIndex == 0){
			if(methodtorun == 1){
				versionString = "sgbm1";

				dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/" +folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";
			}
			if(methodtorun == 2){
				versionString = "sgbm2";

				dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";
			}
			if(methodtorun == 3){
				versionString = "sed";

				dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";
			}
			if(methodtorun == 4){
				versionString = "elas";

				dispImg_string = rootFolder+"/monkaa/disparity_" + versionString + "/"+folder+"/" + zeroPadded(std::to_string(i), 4) + ".pfm";
			}
		}



              if(methodtorun == 0 || (methodtorun > 0 && versionIndex ==0))
                      calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string,dispLevels,false);
          }
        }
			  
      }
    }
  } else if (strcmp(methodName, "flyingthings") == 0) {
  	std::string abcStrings[3] = {"A","B","C"};
	
    for (int abc = 0; abc < 3; abc++) {
      for (int folderNum = 0; folderNum <= 149; folderNum++) {
      	for (int itemNum = 6; itemNum <= 15; itemNum++) {

        for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
          versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
          int dispLevels = 255;

          std::string folders = abcStrings[abc]+"/"+zeroPadded(std::to_string(folderNum), 4);
          std::string base = rootFolder+"/flyingthings3D";
          std::string dispBase = base+"/disparity/TEST/"+folders;
          std::string imgBase = base+"/TEST/"+folders;

		  std::string leftImg_string = imgBase + "/left/" + zeroPadded(std::to_string(itemNum), 4) + ".png";
          std::string rightImg_string = imgBase + "/right/" + zeroPadded(std::to_string(itemNum), 4) + ".png";

		  if(flyingthingsExclude(leftImg_string))
			  continue;
				
          if(isResultsCompMode){
            std::string dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";
            std::string dispImgVis_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + "_vis" +".png";
          
            int sizes[10] = {1,0,0,0};
			
			int nColors = (*sparse)?16:8;
			
					  if(dbCheckMode){
			  if(versionIndex !=0) 
				  continue;
			  printf("%s\n",leftImg_string.c_str());
		  }else{


  			if(methodtorun == 0){	
				  run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,2,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
			}else{
				if(versionIndex == 0){
					if(methodtorun == 1){
						versionString = "sgbm1";

						dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";
						dispImgVis_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + "_vis" +".png";

						printf("python sgbm1.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
					}
					if(methodtorun == 2){
						versionString = "sgbm2";

						dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";
						dispImgVis_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + "_vis" +".png";

						printf("python sgbm2.py %s %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispImgVis_string.c_str(),dispLevels);
					}
					if(methodtorun == 3){
						versionString = "sed";

						dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";

						printf("./SED/experiments/sed_run/sed_run %s %s %s %d\n",leftImg_string.c_str(),rightImg_string.c_str(),dispImg_string.c_str(),dispLevels);
					}
					if(methodtorun == 4){
						versionString = "elas";

						dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";



						std::string rightImg_string_ = rightImg_string+"";
						std::string leftImg_string_ = leftImg_string+"";
	
			
						leftImg_string_ = leftImg_string_.substr(0, leftImg_string_.size()-4)+".pgm";
						rightImg_string_ = rightImg_string_.substr(0, rightImg_string_.size()-4)+".pgm";	

						//printf("convert %s -flatten %s \n",leftImg_string.c_str(),leftImg_string_.c_str());
						//printf("convert %s -flatten %s \n",rightImg_string.c_str(),rightImg_string_.c_str());




						printf("./elas/elas %s %s %s %d\n",leftImg_string_.c_str(),rightImg_string_.c_str(),dispImg_string.c_str(),dispLevels);
					}
				}
			}



            
		  }
          }
          if(isMetricCompMode){
            std::string dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";
            std::string gt_string =  dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) + ".pfm";


		if(versionIndex == 0){
			if(methodtorun == 1){
				versionString = "sgbm1";

				dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";
			}
			if(methodtorun == 2){
				versionString = "sgbm2";

				dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";
			}
			if(methodtorun == 3){
				versionString = "sed";

				dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";
			}
			if(methodtorun == 4){
				versionString = "elas";

				dispImg_string = dispBase +"/left/" + zeroPadded(std::to_string(itemNum), 4) +"_"+ versionString + ".pfm";
			}
		}			


	
	    if(methodtorun == 0 || (methodtorun > 0 && versionIndex ==0))
           	 calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string,dispLevels,false);
          }
        }
			  
      }
    }
  }
}
}



int main(int argc, char * argv[]) {
  char * methodName = argv[1];

  printf("[all,middleburry,kitti2015,realgarden,synthgarden,driving,monkaa,flyingthings] [both,metric,result]\n");

  methodtorun = 0; //atoi(argv[3]);

  if(strcmp(argv[2], "both") == 0){
    isMetricCompMode = true;
    isResultsCompMode = true;
  }else if(strcmp(argv[2], "metric") == 0){
    isMetricCompMode = true;
    isResultsCompMode = false;
  }else if(strcmp(argv[2], "result") == 0){
    isMetricCompMode = false;
    isResultsCompMode = true;
  }

  const char * methodNames[9] = {"middleburry"};
  const char * methodNames2[9] =  {"kitti2015", "realgarden", "synthgarden", "driving","monkaa","flyingthings"};

  if (strcmp(methodName, "all") == 0) {
    for (int i = 0; i < 2; i++) {
      processMethod(methodNames[i]);
    }
  } else if (strcmp(methodName, "all2") == 0) {
    for (int i = 0; i < 5; i++) {
      processMethod(methodNames2[i]);
    }
  } else{
    processMethod(methodName);
  }

  return 0;
}
