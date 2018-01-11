#include "PipelineVLAD.h"

#include <iostream>
#include <fstream>

PipelineVLAD::PipelineVLAD() {
  m_typeImages = "jpeg";
  m_typeMat = "yml";

  m_nameVocabulary = "vocabulary";
  m_nameFeatureDescriptors = "featuresDescriptors";
  m_nameResponseHist = "responseHist";

  m_pathImages = "data/trainimages/";
  m_pathFeatureDescriptors = "data/";
  m_pathVocabulary = "data/vocabulary/";
  m_pathResult = "data/result/";
}

PipelineVLAD::~PipelineVLAD() {
}

void PipelineVLAD::runTrainig() {
  //-- Features descriptors
  cv::Ptr<cv::FeatureDetector> m_detector = new cv::SurfFeatureDetector();
  cv::Ptr<cv::DescriptorExtractor> m_descriptorExtractor = new cv::SurfDescriptorExtractor();
  cv::Mat descriptors, allDescriptors, features;
  std::vector<cv::KeyPoint> keypoints;

  //-- Loading images
  std::vector<std::string> trainImgNames;
  cv::Mat frame;

  trainImgNames = m_dataManagement.loadImageNames(m_pathImages, m_typeImages);

  for (std::size_t i = 0; i < trainImgNames.size(); ++i) {
    frame = cv::imread(trainImgNames[i]);

    if (frame.empty()) {
      std::cerr << "WARNING: Test image not found" << std::endl;
    }

    //-- Feature detection
    // TODO: in eine neue Klasse schreibe
    m_detector->detect(frame, keypoints);
    m_descriptorExtractor->compute(frame, keypoints, descriptors);
    
    allDescriptors.push_back(descriptors);
  }

  //-- Save descriptors
  m_dataManagement.saveMat(m_pathFeatureDescriptors, m_nameFeatureDescriptors, m_typeMat, allDescriptors);


  /***************************************************************
  *       BOW Trainer
  ***************************************************************/
  cv::Mat desc;
  desc = m_dataManagement.loadMat(m_pathFeatureDescriptors, m_nameFeatureDescriptors, m_typeMat);

  cv::Mat trainDesc_32f;
  desc.convertTo(trainDesc_32f, CV_32F);

  cv::BOWKMeansTrainer bowtrainer(200); //INFO: FABMAP: cv::of2::BOWMSCTrainer
  bowtrainer.add(trainDesc_32f);
  
  std::cout << "Clustering Bow features" << std::endl;
  cv::Mat vocabulary = bowtrainer.cluster();
  
  //-- Save vocabulary
  m_dataManagement.saveMat(m_pathVocabulary, m_nameVocabulary, m_typeMat, vocabulary);
 
  /***************************************************************
  *       VLAD
  ***************************************************************/
  //-- Read vocabulary
  cv::Mat vocab; 
  vocab = m_dataManagement.loadMat(m_pathVocabulary, m_nameVocabulary, m_typeMat);

  cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SURF");
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

  //-- Bag-of-Words representation
  cv::BOWImgDescriptorExtractor bide(extractor, matcher);
  bide.setVocabulary(vocab);

  std::vector<cv::KeyPoint> keypoints1;
  cv::Mat desc2, desc1;

  //-- Load images
  std::vector<std::string> imgNames1; 
  imgNames1 = m_dataManagement.loadImageNames(m_pathImages, m_typeImages);
  for (std::size_t i = 0; i < imgNames1.size(); ++i) {
    frame = cv::imread(imgNames1[i]);

    if (frame.empty()) {
      std::cerr << "WARNING: Image not found" << std::endl;
    }

    m_detector->detect(frame, keypoints1);
    m_descriptorExtractor->compute(frame, keypoints1, desc1);

    cv::Mat desc1_32f;
    desc1.convertTo(desc1_32f, CV_32F);

    /********* The new VLAD compute *********/
    std::vector<cv::DMatch> matches;
    matcher->match(desc1_32f, matches); //desc1 contains descriptors for each image

    cv::Mat responseHist(vocabulary.rows, desc1_32f.cols, CV_32FC1, cv::Scalar::all(0.0));


    /*****/
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

    std::cout << "ResponseHist:" << responseVector.rows << "," << responseVector.cols << "," 
      << responseVector.type() << "," << responseVector.channels() << std::endl;
    //cout<<"\n"<<desc.cols;

    if (responseHist.rows != 0) {
      desc2.push_back(responseVector);
    }
  }

  //-- Save resonseHist
  m_dataManagement.saveMat(m_pathFeatureDescriptors, m_nameResponseHist, m_typeMat, desc2);

  /***************************************************************
  *       SVM Train  --> Dont have labels, so no training
  ***************************************************************/
  //-- Loading responseHist
  cv::Mat trainData, labels;
  trainData = m_dataManagement.loadMat(m_pathFeatureDescriptors, m_nameResponseHist, m_typeMat);

  /******/
  int rowCount;
  rowCount = trainData.rows;
  cv::Mat reslabels = cv::Mat::zeros(rowCount, 1, CV_32SC1);
  //reslabels.setTo((float)(count + 1));
  labels.push_back(reslabels);
  /******/

  //-- Set up SVM's parameters
  CvSVMParams params(
    CvSVM::C_SVC, CvSVM::RBF, 1,
    1, 0, 1, 0.1, 0.2, 0, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON));
  //Train array:DD array with responsehistograms values

  cv::Mat svmData;
  trainData.convertTo(svmData, CV_32F);
  //cv::Mat labelData;
  //labelData = labels.t(); //labels: Array of labels


  //-- SVM training:
  //std::cout << "SVM Training!\n";
  //CvSVM svm(svmData, labelData, cv::Mat(), cv::Mat(), params);
  ////params=svm.get_params();
  ////cout<<"\n Svm parameters: "<<svm.get_params();
  //CvParamGrid Cgrid = CvSVM::get_default_grid(CvSVM::C);
  //CvParamGrid gammaGrid = CvSVM::get_default_grid(CvSVM::GAMMA);
  //CvParamGrid pGrid = CvSVM::get_default_grid(CvSVM::P);
  //CvParamGrid nuGrid = CvSVM::get_default_grid(CvSVM::NU);
  //CvParamGrid coeffGrid = CvSVM::get_default_grid(CvSVM::COEF);
  //CvParamGrid degreeGrid = CvSVM::get_default_grid(CvSVM::DEGREE);

  //svm.train_auto(svmData, labelData, cv::Mat(), cv::Mat(), params, 10, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, false);
  //svm.save("classifier_SURF_VLAD_C200_RBF.xml", "SURF");
}

void PipelineVLAD::runTest() {
  //-- Load classifier
  CvSVM svm;
  svm.load("classifier_SURF_VLAD_RBF", "SURF");

  //-- Load vocabulary
  cv::Mat vocabulary;
  vocabulary = m_dataManagement.loadMat("/", "vocabulary_surf_vlad", "xml");

  // INFO: does not work yet
  //_chdir("Images");
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
    
    /*
    int i;
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

void PipelineVLAD::computesDescriptors(cv::Ptr<cv::FeatureDetector>& detector) {

}
