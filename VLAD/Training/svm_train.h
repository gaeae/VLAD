#pragma once

// openCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"

class SVMTraining {
public:
  SVMTraining() {};
  ~SVMTraining() {};

  void train();

private:

};