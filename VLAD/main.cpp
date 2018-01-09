#include "Training\BOWKTrainer.h"
#include "Training\sorter.h"
#include "Training\surf.h"
#include "Training\svm.h"
#include "Training\VLAD.h"

#include "Test\svmTest_VLAD.h"

#include <iostream>
#include <new>
#include <memory>

int main() {
  std::unique_ptr<Sorter> pSorter(new Sorter());
  std::unique_ptr<Surf> pSurf(new Surf());
  std::unique_ptr<BOWKTrainer> pBOWKTrainer(new BOWKTrainer());
  std::unique_ptr<VLAD> pVLAD(new VLAD());
  std::unique_ptr<SVMTraining> pSVMTraining(new SVMTraining());

  std::unique_ptr<SVMTest> pSVMTest(new SVMTest());

  //-- Training
  pSorter->sort();
  pSurf->surf();
  pBOWKTrainer->createBOWDescriptors();
  pVLAD->bowKMeansRecogniser();
  pSVMTraining->train();

  //-- Test
  pSVMTest->test();


}