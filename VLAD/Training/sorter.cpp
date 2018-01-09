//Sorts images as per labels
#include "sorter.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>


void Sorter::sort()
{
    std::ifstream f1("labels.txt");
    
    //chdir("Images");
   // system("ls");
    std::string str;
    std::string label;
  //  system("pwd");
    while( !f1.eof() )
    {   
        
        f1 >> str;
        f1 >> label;
        
        
        std::string command = std::string("mv Images/")  + str + std::string(" ./") + label + std::string("/") + str;
        std::cout<<"\n" << command;
        system(command.c_str());
        
    }
    
    //return 0;
}
