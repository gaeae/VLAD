//BOWKmeans training- Creates BOW Descriptors

#include "BOWKTrainer.h"
//#include <highgui.h>
#include <opencv\highgui.h>
#include "opencv2/opencv.hpp"
#include <string>
//#include "trainer.hpp"
#include <iostream>
#include <fstream>
//#include "HarrisDetector.cpp"

#include <direct.h>


using namespace cv;
using namespace std;

void BOWKTrainer::createBOWDescriptors() {
  //map<string,Mat> class_data;

  string fname;
  Mat trainDesc;
  _chdir("SURFPCA");
  ifstream f1;
  f1.open("list.txt");

  //char opt = 'y';
  while (!f1.eof()) {
    //cout<<"\n Enter filename";
    //cin>>fname;
    f1 >> fname;
    cout << "Adding descriptors from " << fname;
    FileStorage fs2(fname, FileStorage::READ);
    Mat add1;
    fs2["desc_all"] >> add1;
    cout << "\n" << add1.rows;
    trainDesc.push_back(add1);

    //cout<<"\n Add another file?(y/n)";
    //cin>>opt;
  }
  cout << "\n Train desc" << trainDesc.rows;
  Mat trainDesc_32f;
  trainDesc.convertTo(trainDesc_32f, CV_32F);
  BOWKMeansTrainer bowtrainer(200);
  bowtrainer.add(trainDesc_32f);
  cout << "\n clustering Bow features" << endl;

  Mat vocabulary = bowtrainer.cluster();
  _chdir("..");
  FileStorage fs1("vocabulary_surfpca_vlad_200.xml", FileStorage::WRITE);

  fs1 << "vocabulary" << vocabulary;

  fs1.release();
  f1.close();
  //return 0;
}
