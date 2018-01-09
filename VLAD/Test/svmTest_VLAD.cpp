//SVM test
#include "svmTest_VLAD.h"

#include <direct.h>
#include <fstream>
#include <iostream>
#include <string>

//#include "HarrisDetector.cpp"
//#include "highDensity.cpp"

void  SVMTest::test() {

  //Test
  CvSVM svm;
  svm.load("classifier_SURF_VLADNU_LINEAR.xml", "SURF");
  //Mat testImg = imread("Test");

  cv::Mat vocabulary;

  cv::FileStorage fs3("vocabulary_surf_vlad.xml", cv::FileStorage::READ);
  std::cout << "Reading Vocabulary from file";

  fs3["vocabulary"] >> vocabulary;
  fs3.release();
  _chdir("Images");
  std::ifstream f1;
  f1.open("list.txt");
  std::ofstream f2("output_SURFVLAD_Linear.txt");
  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SURF");

  cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SURF");
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
  cv::BOWImgDescriptorExtractor bowide(extractor, matcher);
  bowide.setVocabulary(vocabulary);
  while (!f1.eof()) {
    cv::string imgName;
    f1 >> imgName;
    std::cout << "\n" << imgName;
    cv::Mat img = cv::imread(imgName);
    if (!img.data)
      continue;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat desc1(0, 30, CV_64FC1);
    //Mat responseHist;
    //Mat responseHist(1, 1000, CV_32FC1, Scalar::all(0.0));
    cvtColor(img, img, CV_BGR2GRAY);
    cv::SURF surf;
    cv::vector<cv::Point2f> points;
    //harrisFeatures(img, points);
    //cout<<"\n POints size:"<<points.size();
    int i;
    /*
    for(i=0;i<points.size();i++)
    {
            KeyPoint temp(points[i],10,-1,0,0,-1);
            keypoints.push_back(temp);
            //cout<<"\n Point "<<i<<" "<< keypoints[i].pt.x <<" "<<keypoints[i].pt.y;
    }
    */
    surf(img, img, keypoints, desc1, false);
    //bowide.compute(img,keypoints,responseHist);
    cv::Mat desc1_32f;
    desc1.convertTo(desc1_32f, CV_32F);
    /* **********The new VLAD.compute  **********/

    cv::vector<cv::DMatch> matches;
    matcher->match(desc1_32f, matches); //desc1 contains descriptors for each image


    cv::Mat responseHist(vocabulary.rows, desc1_32f.cols, CV_32FC1, cv::Scalar::all(0.0));
    // float *dptr = (float*)responseHist.data;
    for (size_t i = 0; i < matches.size(); i++) {
      int queryIdx = matches[i].queryIdx;
      int trainIdx = matches[i].trainIdx; // cluster index
      CV_Assert(queryIdx == (int)i);
      cv::Mat residual;

      subtract(desc1_32f.row(matches[i].queryIdx), vocabulary.row(matches[i].trainIdx), residual, cv::noArray(), CV_32F);
      add(responseHist.row(matches[i].trainIdx), residual, responseHist.row(matches[i].trainIdx), cv::noArray(), responseHist.type());



    }
    responseHist /= norm(responseHist, cv::NORM_L2, cv::noArray());

    cv::Mat responseVector(1, vocabulary.rows*desc1_32f.cols, CV_32FC1, cv::Scalar::all(0.0));

    responseVector = responseHist.reshape(0, 1);


    float predLabel = 4;
    if (responseVector.rows != 0) {
      predLabel = svm.predict(responseVector, false);
    }
    std::cout << "\n value:" << predLabel;
    f2 << imgName << " " << predLabel << "\n";
  }

  f2.close();
  //return 0;
}
