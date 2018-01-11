#pragma once

// own
/// Training
#include "Training\BOWKTrainer.h"
#include "Training\sorter.h"
#include "Training\surf.h"
#include "Training\svm_train.h"
#include "Training\VLAD.h"
/// Test
#include "Test\svmTest_VLAD.h"
/// anything else
#include "DataManagement.h"

// openCV
#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>

// STL
#include <memory>

class PipelineVLAD {
public:
  PipelineVLAD();
  ~PipelineVLAD();

  void runTrainig();
  void runTest();

private:

  void computesDescriptors(cv::Ptr<cv::FeatureDetector> &detector);

  
  VLAD m_Vlad;

  std::string m_typeMat;
  std::string m_typeImages;

  std::string m_nameVocabulary;
  std::string m_nameFeatureDescriptors;
  std::string m_nameResponseHist;

  std::string m_pathImages;
  std::string m_pathVocabulary;
  std::string m_pathFeatureDescriptors;
  std::string m_pathResult;
};