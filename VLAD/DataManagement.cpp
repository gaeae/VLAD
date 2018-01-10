#include "DataManagement.h"

std::vector<std::string> DataManagement::loadImageNames(std::string pathImages, std::string dataType) {
  pathImages = pathImages + "*." + dataType;
  std::cout << "Loading images from:\n   " + pathImages << std::endl;

  cv::Mat frame;
  cv::String folderpath = pathImages;

  if (folderpath.empty()) {
    std::cerr << "ERROR: Following path doesn't exist: " + folderpath;
  }

  std::vector<cv::String> filenames;
  cv::glob(folderpath, filenames, false);
  std::vector<std::string> imageNames;

  for (size_t i = 0; i < filenames.size(); ++i) {
    imageNames.push_back(filenames[i]);
    //std::cout << " " << std::endl;
    //std::cout << "Datei Nummer: " << i + 1 << std::endl << "File: " << filenames[i] << std::endl;
  }
  return imageNames;
}

cv::VideoCapture DataManagement::loadMovie(std::string pathMovie, std::string name, std::string dataType) {
  pathMovie = pathMovie + name + ".avi";
  std::cout << "Loading movie " + name + " from:\n   " << pathMovie << std::endl;

  cv::VideoCapture movie;
  movie.open(pathMovie);
  if (!movie.isOpened()) {
    std::cerr << "ERROR: " << pathMovie << ": training movie not found" << std::endl;
  }
  return movie;
}

void DataManagement::saveMat(std::string path, std::string name, std::string dataType, cv::Mat mat) {
  std::cout << "Saving:   " << name << " ... ";

  if (path.at(path.length() - 1) != '/') {
    path = path + "/";
  }

  cv::FileStorage fs;
  fs.open(path + name + "." + dataType, cv::FileStorage::WRITE);
  fs << name << mat;
  fs.release();

  std::cout << " saved!" << std::endl;
}

cv::Mat DataManagement::loadMat(std::string path, std::string name, std::string dataType) {

  std::cout << "Loading:    " + name << std::endl;

  if (path.at(path.length() - 1) != '/') {
    path = path + "/";
  }

  cv::FileStorage fs;
  cv::Mat temp;

  fs.open(path + name + "." + dataType, cv::FileStorage::READ);
  fs[name] >> temp;
  if (temp.empty()) {
    std::cerr << "ERROR: " + name + " not found in\n  " + path << std::endl;
  }
  fs.release();

  return temp;
}