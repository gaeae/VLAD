//BOWKmeans training- Creates BOW Descriptors
#include "BOWKTrainer.h"

#include <direct.h>
#include <fstream>
#include <iostream>
#include <string>
//#include "trainer.hpp"
//#include "HarrisDetector.cpp"


void BOWKTrainer::createBOWDescriptors() {
  //map<string,Mat> class_data;

  std::string fname;
  cv::Mat trainDesc;
  _chdir("SURFPCA");
  std::ifstream f1;
  f1.open("list.txt");

  //char opt = 'y';
  while (!f1.eof()) {
    //cout<<"\n Enter filename";
    //cin>>fname;
    f1 >> fname;
    std::cout << "Adding descriptors from " << fname;
    cv::FileStorage fs2(fname, cv::FileStorage::READ);
    cv::Mat add1;
    fs2["desc_all"] >> add1;
    std::cout << "\n" << add1.rows;
    trainDesc.push_back(add1);

    //cout<<"\n Add another file?(y/n)";
    //cin>>opt;
  }
  std::cout << "\n Train desc" << trainDesc.rows;
  cv::Mat trainDesc_32f;
  trainDesc.convertTo(trainDesc_32f, CV_32F);
  cv::BOWKMeansTrainer bowtrainer(200); //INFO: gleiche wie in FABMAP
  bowtrainer.add(trainDesc_32f);
  std::cout << "\n clustering Bow features" << std::endl;

  cv::Mat vocabulary = bowtrainer.cluster();
  _chdir("..");
  cv::FileStorage fs1("vocabulary_surfpca_vlad_200.xml", cv::FileStorage::WRITE);

  fs1 << "vocabulary" << vocabulary;

  fs1.release();
  f1.close();
  //return 0;
}