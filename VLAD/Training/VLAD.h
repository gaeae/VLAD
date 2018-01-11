#pragma once

// openCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"

// own
#include "../DataManagement.h"

class VLAD {
public:
  VLAD();
  ~VLAD() {};

  void bowKMeansRecogniser();

private:

  DataManagement m_dataManagement; 
};