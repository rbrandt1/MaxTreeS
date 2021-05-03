#include "core.hpp"
#include "opencv_pfm.hpp"
#include <cstdlib>
#include <chrono>
#include <thread>
#include<iostream>
#include <cstdlib>

bool isMetricCompMode;
bool isResultsCompMode;

std::string rootFolder = "./datasets";

bool ends_with(std::string  const & value, std::string  const & ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void writeDispmap(cv::Mat img, std::string str) {
	img.convertTo(img, CV_32F);
	if(!ends_with(str, ".pfm")){
		  for(int r =0;r<img.rows;r++){
		for(int c =0;c<img.cols;c++){
		  if(img.at<float>(r,c) == -1){
			  img.at<float>(r,c) = 0;
		  };
		}
	  }
	  cv::imwrite(str.c_str(), img);
	}else{
		for(int r =0;r<img.rows;r++){
			for(int c =0;c<img.cols;c++){
			  if(img.at<float>(r,c) == -1){
				  img.at<float>(r,c) = std::numeric_limits<float>::infinity();
			  };
			}
		}
		  
		opencv_pfm::imwrite_pfm(str.c_str(), img,1,1);
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


void writeVisError(cv::Mat gt, cv::Mat dispMap,int dispLevels){

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



void run(bool sparse, std::string methodName,std::string versionString,std::string resultsFile_String, std::string leftImg_string,std::string rightImg_string,std::string dispImg_string,std::string dispImgVis_string,int disparityLevels,int factor,float alpha,int minAreaMatchedFineTopNode,int nColors,int * sizes,int total,uint kernelCostComp,int kernelCostVolume,int everyNthRow, float minConfidencePercentage,float allowanceSearchRange){

	cv::Mat img_l = cv::imread(leftImg_string.c_str());
	cv::Mat img_r = cv::imread(rightImg_string.c_str());

	int maxAreaMatchedFineTopNode = img_l.cols / factor;

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


void calcMetricsMasked(std::string methodName,std::string versionString,std::string resultsFile_String, std::string dispImg_string,std::string gt_string,int dispLevels,std::string mask_string){
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

	if (dmap.cols > 0 && gt.cols > 0){
		
		if (dmap.channels() > 1){
			cv::cvtColor(dmap, dmap, cv::COLOR_RGB2GRAY);
		}
		if (gt.channels() > 1){
			cv::cvtColor(gt, gt, cv::COLOR_RGB2GRAY);
		}	
		
		dmap.convertTo(dmap,CV_32FC1);
		gt.convertTo(gt,CV_32FC1);
		

		// synthgarden
		if(strcmp(methodName.c_str(), "synthgarden") == 0){
		  for(int r =0;r<gt.rows;r++){
			for(int c =0;c<gt.cols;c++){
			  float val = gt.at<float>(r,c);
				if(val == 0){
					gt.at<float>(r,c) = -1;
				}
			} 
		  }
		}
		
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
			

		}
	

		// our maps invalid
		for(int r =0;r<dmap.rows;r++){
			for(int c =0;c<dmap.cols;c++){
				if(dmap.at<float>(r,c) <= 0 || dmap.at<float>(r,c) > 100000){
					dmap.at<float>(r,c) = -1;
				}
			 }
		 }
		
		float avgerr = metric_avgerr(dmap, gt, dispLevels,true);
		float rmse = metric_rmse(dmap, gt, dispLevels,true);
		float bad1 = metric_bad(dmap, gt, 1, dispLevels,true);
		float bad2 = metric_bad(dmap, gt, 2, dispLevels,true);
		float bad3 = metric_bad(dmap, gt, 3, dispLevels,true);
		float bad4 = metric_bad(dmap, gt, 4, dispLevels,true);
		float bad5 = metric_bad(dmap, gt, 5, dispLevels,true);	
		
		
		float avgerr_clipped = metric_avgerr(dmap, gt, dispLevels,false);
		float rmse_clipped = metric_rmse(dmap, gt, dispLevels,false);
		float bad1_clipped = metric_bad(dmap, gt, 1, dispLevels,false);
		float bad2_clipped = metric_bad(dmap, gt, 2, dispLevels,false);
		float bad3_clipped = metric_bad(dmap, gt, 3, dispLevels,false);
		float bad4_clipped = metric_bad(dmap, gt, 4, dispLevels,false);
		float bad5_clipped = metric_bad(dmap, gt, 5, dispLevels,false);	
		
		float D1ALL = metric_D1ALL(dmap, gt); 	
		float density = metric_density(dmap, gt);     
		
		writeInfoResults(resultsFile_String, methodName, versionString, resultsFile_String, dispImg_string, avgerr, rmse, bad1, bad2, bad3, bad4, bad5,D1ALL,density,avgerr_clipped, rmse_clipped, bad1_clipped, bad2_clipped, bad3_clipped, bad4_clipped, bad5_clipped);
		
		if(false){
			writeVisError(gt, dmap,dispLevels);
		}
	}
}

void calcMetrics(std::string methodName,std::string versionString,std::string resultsFile_String, std::string dispImg_string,std::string gt_string,int dispLevels){
	calcMetricsMasked( methodName, versionString, resultsFile_String,  dispImg_string, gt_string, dispLevels, "notmasked");
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

    return versionString0+versionString1+versionString2;
}

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
      if (strcmp(methodName, "middlebury") == 0) {
        resultsFile_String = rootFolder+"/middlebury/results.csv";
        initializeResultsFile(resultsFile_String);
      } else if (strcmp(methodName, "synthgarden") == 0) {
        resultsFile_String = rootFolder+"/synthgarden/results.csv";
        initializeResultsFile(resultsFile_String);
      } else if (strcmp(methodName, "kitti2015") == 0) {
        resultsFile_String = rootFolder+"/kitti2015/results.csv";
        initializeResultsFile(resultsFile_String);
      }
  }
  if(isMetricCompMode){
      if (strcmp(methodName, "middlebury") == 0) {
        resultsFileAccuracy_String = rootFolder+"/middlebury/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      } else if (strcmp(methodName, "synthgarden") == 0) {
        resultsFileAccuracy_String = rootFolder+"/synthgarden/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      } else if (strcmp(methodName, "kitti2015") == 0) {
        resultsFileAccuracy_String = rootFolder+"/kitti2015/resultsAccuracy.csv";
        initializeResultsAccuracyFile(resultsFileAccuracy_String);
      }
  }

  if (strcmp(methodName, "middlebury") == 0) {
    std::string folders[15] = {"Adirondack", "ArtL", "Jadeplant", "Motorcycle", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "Playtable", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"}; 
    int disps[15] = {290, 256, 640, 280, 280, 260, 260, 300, 330, 290, 290, 260, 240, 256, 760 }; 

    for (int i = 0; i < 15; i++) {
      std::string folder = folders[i];
      for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
		
        versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
        int dispLevels = disps[i];

        if(isResultsCompMode){
          std::string leftImg_string = rootFolder+"/middlebury/trainingF/" + folder + "/im0.png";
          std::string rightImg_string = rootFolder+"/middlebury/trainingF/" + folder + "/im1.png";
          std::string dispImg_string = rootFolder+"/middlebury/trainingF/" + folder + "/disp0Mt_" + versionString + ".pfm";
          std::string dispImgVis_string = rootFolder+"/middlebury/trainingF/" + folder + "/disp0Mt_" + versionString + ".png";
          int sizes[5] = {1,0,0,0};
		
		  int nColors = (*sparse)?16:8;
		  		
          run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,3,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
        }
        if(isMetricCompMode){
          std::string dispImg_string = rootFolder+"/middlebury/trainingF/" + folder + "/disp0Mt_" + versionString + ".pfm";
	  
          //dispImg_string = rootFolder+"/middlebury/trainingF/" + folder + "/sed.png";

          std::string gt_string = rootFolder+"/middlebury/trainingF/" + folder + "/disp0GT.pfm";
          calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string, dispLevels);
        }
      }
	  
    }
  } else if (strcmp(methodName, "synthgarden") == 0) {

	std::string weather[20] = {"clear_0001","clear_0128","clear_0160","clear_0224","cloudy_0001","cloudy_0128","cloudy_0160","cloudy_0224","overcast_0001","overcast_0128","overcast_0160","overcast_0224","sunset_0001","sunset_0128","sunset_0160","sunset_0224","twilight_0001","twilight_0128","twilight_0160","twilight_0224"};

        for (int k = 0; k < 20; k += 1) {
	 for (int imgID = 1; imgID <= 100; imgID += 1) {
            for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){
         		versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
           		int dispLevels = 100;

			std::string numStr;
			if(imgID < 10){
				numStr = "0000"+std::to_string(imgID);
			}else if(imgID < 100){
				numStr = "000"+std::to_string(imgID);
			}else{
				numStr = "00"+std::to_string(imgID);
			}

                if(isResultsCompMode){
			

			std::string leftImg_string = rootFolder+"/synthgarden/"+weather[k]+"/vcam_0/vcam_0_f"+ numStr + "_undist.png";
			std::string rightImg_string = rootFolder+"/synthgarden/"+weather[k]+"/vcam_1/vcam_1_f"+ numStr + "_undist.png";
			std::string dispImg_string =  rootFolder+"/synthgarden/"+weather[k]+"/vcam_0/vcam_0_f"+numStr + "_disp_" + versionString + ".pfm";
			std::string dispImgVis_string = rootFolder+"/synthgarden/"+weather[k]+"/vcam_0/vcam_0_f"+ numStr + "_dispVis_" + versionString + ".png";
			int sizes[5] = {1,0,0,0};

			int nColors = (*sparse)?16:8;

			run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,15,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
                
			cv::Mat dispMap = opencv_pfm::imread_pfm(dispImg_string.c_str());
		
			dispMap.convertTo(dispMap, CV_32FC1);
			float f = 640;
			float b = 0.03;
			for (int x = 0; x < 640; x++) {
				for (int y = 0; y < 480; y++) {
					if(dispMap.at<float>(y, x) <= 0){
						dispMap.at<float>(y, x) = -1;
					}else{
						dispMap.at<float>(y, x) = ((f*b)/dispMap.at<float>(y, x));
					}
				}
			}
		

			cv::flip(dispMap, dispMap, 0);

			writeDispmap(dispMap, dispImg_string);
			writeVisDispmap(dispMap, dispImgVis_string, dispLevels);
					
		}
                if(isMetricCompMode){
			std::string dispImg_string =  rootFolder+"/synthgarden/"+weather[k]+"/vcam_0/vcam_0_f"+numStr + "_disp_" + versionString + ".pfm";
			std::string gt_string =  rootFolder+"/synthgarden/"+weather[k]+"/vcam_0/vcam_0_f"+numStr + "_dmap.bin";
			std::string gt_string2 =  rootFolder+"/synthgarden/"+weather[k]+"/vcam_0/vcam_0_f"+numStr + "_dmap.pfm";
			std::string gt_string3 =  rootFolder+"/synthgarden/"+weather[k]+"/vcam_0/vcam_0_f"+numStr + "_dmap_vis.png";

			std::ifstream inputfile(gt_string.c_str(), std::ifstream::binary);

			cv::Mat gt = cv::Mat(480, 640, CV_32F);

		
			for (int x = 0; x < 640; x++) {
				for (int y = 0; y < 480; y++) {

					unsigned char temp[sizeof(float)];
					inputfile.read(reinterpret_cast<char*>(temp),
							sizeof(float));
					unsigned char t = temp[0];
					temp[0] = temp[3];
					temp[3] = t;
					t = temp[1];
					temp[1] = temp[2];
					temp[2] = t;

					gt.at<float>(y, x) = ((float) reinterpret_cast<float&>(temp));
				}
			}
				
			writeDispmap(gt, gt_string2);
			writeVisDispmap(gt, gt_string3, dispLevels);
			
                  	calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string2,dispLevels);
                }
		}
            }
			
        }
    
  } else if (strcmp(methodName, "kitti2015") == 0) {
   
    for (int k =0; k < 200; k += 1) {
      for(int versionIndex = 0 ; versionIndex < 2;versionIndex++){

        versionString = genVersionString(versionIndex, useOldGradCost, sparse, useBilateral);
 	    mkdir2(rootFolder+"/kitti2015/training/disp_" + versionString);
        int dispLevels = 255;
        if(isResultsCompMode){
          std::string leftImg_string = rootFolder+"/kitti2015/training/image_2/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          std::string rightImg_string = rootFolder+"/kitti2015/training/image_3/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          std::string dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          std::string dispImgVis_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/visualization_" + zeroPadded(std::to_string(k), 6) + "_10.png";
          int sizes[5] = {1,0,0,0};
		  
		  int nColors = (*sparse)?16:8;
		  
          run(* sparse, methodName, versionString, resultsFile_String,  leftImg_string,rightImg_string,dispImg_string,dispImgVis_string,dispLevels,3,.8,0,nColors,sizes,4,10,21,1,minConfidencePercentage,allowanceSearchRange);
        }
        if(isMetricCompMode){
          std::string dispImg_string = rootFolder+"/kitti2015/training/disp_" + versionString + "/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          std::string gt_string = rootFolder+"/kitti2015/training/disp_noc_0/" + zeroPadded(std::to_string(k), 6) + "_10.png";
          calcMetrics(methodName,versionString,resultsFileAccuracy_String, dispImg_string,gt_string,dispLevels);
        }
      }  
	  	  
    }
  }
}



int main(int argc, char * argv[]) {
  char * methodName = argv[1];

  printf("[all,middlebury,kitti2015,synthgarden] [both,metric,result]\n");


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

  const char * methodNames[9] = {"middlebury", "kitti2015", "synthgarden"};

  if (strcmp(methodName, "all") == 0) {
    for (int i = 0; i < 3; i++) {
      processMethod(methodNames[i]);
    }
  } else{
    processMethod(methodName);
  }

  return 0;
}
