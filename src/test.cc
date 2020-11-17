#include <vector>
#include <algorithm> /* for random_shuffle */
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

int main(int argc, char **argv)
{
   string file_name="arwa";
   string file_name2=file_name + "_z_";
   std::stringstream sstm;
   sstm << file_name2 << 1;
   string result = sstm.str();
   cout<<file_name;

   ofstream fs;
   fs.open(result.c_str(), ios::out);
   //myfile.open (file_name, ios::out);
   
   for( int i=0; i<10; i++){
	for( int j=0; j<10; j++){
		//cout<<desc_for_nndes[i][j]<<" ";
		fs <<i<<j;
	}
	//cout<<"\n";
	fs <<"\n";
    }
    fs.close();
return 0;
}
