//BOWKMeans Recogniser
#include "VLAD.h"

#include <direct.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

//#include "HarrisDetector.cpp"


void VLAD::bowKMeansRecogniser() {
  cv::Mat vocabulary;

  std::cout << "Reading Vocabulary from file";
  cv::FileStorage fs1("vocabulary_surf_vlad.xml", cv::FileStorage::READ);

  fs1["vocabulary"] >> vocabulary;
  fs1.release();
  cv::Mat img;
  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SURF");

  cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SURF");
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
  cv::BOWImgDescriptorExtractor bowide(extractor, matcher);
  bowide.setVocabulary(vocabulary);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat desc, desc1;
  //vector<Mat> desc;
  //Setting up training data
  char opt = 'y';
  int i = 0;
  char compName[50], response[50];
  std::ifstream specNameList;
  specNameList.open("dir.txt");
  while (!specNameList.eof()) {
    //cout<<"\n Enter Species for training";
    //cin>>compName;
    specNameList >> compName;
    strcpy(response, compName);

    strcat(response, "SURFVLADResponse.xml");
    std::cout << "\n Adding to " << response;
    _chdir(compName);


    std::string imgName;
    std::ifstream f1;
    f1.open("list.txt");

    cv::FileStorage fs2(response, cv::FileStorage::WRITE);
    //int flag =0;
    cv::Mat desc;
    //Mat responseCopy;
    while (!f1.eof()) {
      //Mat responseHist(1,30,CV_32FC1);
      f1 >> imgName;
      std::cout << "\n" << imgName;
      img = cv::imread(imgName);
      if (!img.data)
        continue;
      cvtColor(img, img, CV_BGR2GRAY);
      cv::SURF surf;
      std::vector<cv::Point2f> points;
      //harrisFeatures(img, points);
      //cout<<"\n POints size:"<<points.size();
      /*

      for(i=0;i<points.size();i++)
      {
          KeyPoint temp(points[i],10,-1,0,0,-1);
          keypoints.push_back(temp);
          //cout<<"\n Point "<<i<<" "<< keypoints[i].pt.x <<" "<<keypoints[i].pt.y;
      }
      */
      surf(img, img, keypoints, desc1, false);
      cv::Mat desc1_32f;
      desc1.convertTo(desc1_32f, CV_32F);
      /* **********The new VLAD.compute  **********/

      std::vector<cv::DMatch> matches;
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

      int i;
      std::cout << "\n ResponseHist:" << responseVector.rows << "," << responseVector.cols << "," << responseVector.type() << "," << responseVector.channels();
      //cout<<"\n"<<desc.cols;

      if (responseHist.rows != 0) {

        desc.push_back(responseVector);

      }
      //cout<<"\n "<<desc.rows<<" "<<desc.cols;

    }

    fs2 << "responseHist" << desc;

    fs2.release();

    f1.close();
    //f2.close();

    //cout<<"\nAnother comp?(y/n)";
    //cin>>opt;
    _chdir("..");

  }
  //return 0;
}