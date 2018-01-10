#include "surf.h"

#include <direct.h>
#include <fstream>
#include <string>

//#include "HarrisDetector.cpp"


void  Surf::surf() {
  cv::Mat img;
  cv::Mat threshImg;

  //adaptiveThreshold(img, threshImg);
  std::ifstream dir_list;
  dir_list.open("dir.txt");
  if (!dir_list.is_open()) {
    std::cout << "file list doesnot exist";
    exit(0);
  }
  std::ifstream f1;


  char dirName[20];
  std::string imgName;
  while (!dir_list.eof()) {
    cv::Mat desc_all;
    dir_list >> dirName;
    std::cout << "\n Creating feature vectors for " << dirName;
    _chdir(dirName);
    f1.open("list.txt");
    if (f1.is_open() == 0) {
      std::cout << "\n Input File does not exist for" << dirName << "!";
      exit(0);
    }
    while (!f1.eof()) {
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat desc;
      f1 >> imgName;
      std::cout << "\n" << imgName;
      img = cv::imread(imgName);
      if (!img.data) {
        std::cout << "\n File not found!";
        continue;
      }
      cvtColor(img, img, CV_BGR2GRAY);
      cv::SURF surf;

      /*
      std::vector<cv::Point2f> points;
      //harrisFeatures(img, points);
      //std::cout << "\n POints size:" << points.size();
      //int i;
      
      for(i=0;i<points.size();i++)
      {
          KeyPoint temp(points[i],10,-1,0,0,-1);
          keypoints.push_back(temp);
          cout<<"\n Point "<<i<<" "<< keypoints[i].pt.x <<" "<<keypoints[i].pt.y;
      }
      */

      surf(img, img, keypoints, desc, false); ///finds keypoint and compute descriptors
      desc_all.push_back(desc);

    }
    std::string featureVectorFile = std::string(dirName) + std::string("SURF.xml");
    std::cout << "\n Writing Feature vectors into: " << featureVectorFile;
    _chdir("../SIFT");
    cv::FileStorage fs(featureVectorFile, cv::FileStorage::WRITE);

    fs << "desc_all" << desc_all;

    fs.release();
    f1.close();
    _chdir("..");
  }
  dir_list.close();

  //return 1;
}