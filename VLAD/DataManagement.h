#pragma once

// openCV
#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>

// STL
#include <vector>

/*!
\class DataManagement
\brief Loading images or video files. Depending what you need. 
*/
class DataManagement {
public:

  DataManagement() {};
  ~DataManagement() {};

  /*!
  \brief Loading images names and returns a vector of strings
  \param path Path containing the images
  \param dataType Specifies the images data type like jpeg
  \retval Returns a vector containing the names of the images
  */
  std::vector<std::string> loadImageNames(std::string pathImages, std::string dataType);

  /*!
  \brief Loading movie / video and returns a VideoCapture format
  \param path Path containing the movie / video
  \param name Name of the video
  \param dataType Specifies the data type like avi
  \retval Returns an VideoCapture
  */
  cv::VideoCapture loadMovie(std::string pathMovie, std::string name, std::string dataType);


  /*!
  \brief Save a Matrix
  \param path Path to save the matrix
  \param name Name of the saving matrix, also in the yaml
  \param dataType Type of the data like yml or xml
  \param mat Matrix that is to be saved
  */
  void saveMat(std::string path, std::string name, std::string dataType, cv::Mat mat);


  /*!
  \brief Load a yml or xml Matrix and return cv::Mat
  \param path Path to load the matrix
  \param name Name of the loading matrix
  \param dataType Type of the data like yml or xml
  */
  cv::Mat loadMat(std::string path, std::string name, std::string dataType);

private:

};