#include "PipelineVLAD.h"


PipelineVLAD::PipelineVLAD() {
  m_pathImages = "data";
  m_typeImages = "jpeg";

  m_typeMat = "yml";
}

PipelineVLAD::~PipelineVLAD() {
}

void PipelineVLAD::runTrainig() {
  //-- Features descriptors
  cv::SURF surf; // INFO: für jedes Frame ein neues Objekt?
  cv::Mat descriptors, allDescriptors, features;
  std::vector<cv::KeyPoint> keypoints;

  //-- Loading images
  std::vector<std::string> imgNames;
  std::vector<std::string> trainImgNames;
  cv::Mat frame;

  DataManagement loadData;
  imgNames = loadData.loadImageNames(m_pathImages, m_typeImages);
  loadData.~DataManagement(); // INFO

  for (std::size_t i = 0; i < trainImgNames.size(); ++i) {
    frame = cv::imread(trainImgNames[i]);

    if (frame.empty()) {
      std::cerr << "WARNING: Test image not found" << std::endl;
    }

    //-- Feature detection
    // TODO: in eine neue Klasse schreibe
    // INFO: wo liegt der Unterschied zu BOW Algorithmus?
    cv::cvtColor(frame, frame, CV_BGR2GRAY);
    surf(frame, frame, keypoints, descriptors);
    
    allDescriptors.push_back(descriptors);
  }

  //-- Save descriptors
  m_dataManagement.saveMat("", "featuresDescriptors", m_typeMat, allDescriptors);


  /***************************************************************
  *       BOW Trainer
  ***************************************************************/
  cv::Mat desc;
  desc = m_dataManagement.loadMat("", "featuresDescriptors", m_typeMat);

  cv::Mat trainDesc_32f;
  desc.convertTo(trainDesc_32f, CV_32F);

  cv::BOWKMeansTrainer bowtrainer(200); //INFO: FABMAP: cv::of2::BOWMSCTrainer
  bowtrainer.add(trainDesc_32f);
  
  std::cout << "Clustering Bow features" << std::endl;
  cv::Mat vocabulary = bowtrainer.cluster();
  
  //-- Save vocabulary
  m_dataManagement.saveMat("", "Vocabulary", m_typeMat, vocabulary);
 
  /***************************************************************
  *       VLAD
  ***************************************************************/
}

void PipelineVLAD::runTest() {

}

void PipelineVLAD::computesDescriptors(cv::Ptr<cv::FeatureDetector>& detector) {

}
