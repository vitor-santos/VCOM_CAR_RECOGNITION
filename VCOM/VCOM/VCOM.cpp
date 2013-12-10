#include "stdafx.h"
#include <opencv/cv.h>
#include <fstream>
#include <iostream>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;


string convertInt(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

bool openImage(const string &f, Mat &image, int mode)
{
	string filename="C:\\Dataset\\cars\\"+f+".image.png";
	if (mode==1)
		image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE );
	else
		image=imread(filename);
	if( !image.data ) {
		cout << " --(!) Error reading image " << filename << endl;
		return false;
	}
	return true;
}


int main( int argc, char** argv ) 
{
	vector<KeyPoint> keypoints;
	Mat descriptors;
	Mat dictionary;
	vector<vector<KeyPoint>> results;
	int d,m;
	Ptr<DescriptorExtractor > extractor;
	Ptr<FeatureDetector> detector;

	//verficar se existem dados em memória e ler do ficheiro
	
	cout<<"Insert the number corresponding to the desired FeatureDetector algorithm:"<<endl<<"1 - SIFT"<<endl<<"2 - SURF"<<endl; 
	cin>>d;
	cout<<"Insert the number corresponding to the desired DescriptorMatcher algorithm:"<<endl<<"1 - FlannBased"<<endl<<"2 - BruteForce"<<endl;
	cin>>m;
	
	
	
	if(d==1)
	{
		extractor=Ptr<DescriptorExtractor>(new SiftDescriptorExtractor());
		detector = FeatureDetector::create("SIFT");
	}
	else 
	{
		extractor=Ptr<DescriptorExtractor>(new SurfDescriptorExtractor());
		detector = FeatureDetector::create("SURF");
	}

	string mat;
	if(m==1)
		mat="FlannBased";
	else
		mat="BruteForce";

	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());	
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(mat);

	//Initial training and clustering
	try
	{
		//open the file with the training images
		ifstream infile("C:\\Dataset\\cars_test.txt");
		string line;
		Mat image;
		int i=0;
		
		while(getline(infile,line))
		{
			//cout<<line<<endl;
			//try to read image with the corresponding filename. We want to read in grayscale so the descriptors are color invariant.
			openImage(line,image,1);	

			detector->detect(image,keypoints);
			extractor->compute(image,keypoints,descriptors);    
			training_descriptors.push_back(descriptors);
			cout<<i<<endl;
			i++;
		}

		cout<<"Total descriptors: "<<training_descriptors.rows<<endl;

		BOWKMeansTrainer bowTrainer(100, TermCriteria(), 1, KMEANS_PP_CENTERS);
		BOWImgDescriptorExtractor bowExtractor(detector, matcher);

		bowTrainer.add(training_descriptors);

		dictionary = bowTrainer.cluster();
		bowExtractor.setVocabulary(dictionary);

		//Aqui começa a segunda parte do treino
		ifstream infile2("C:\\Dataset\\cars_test.txt");
		string line2;
		
		//Nesta altura é melhor guardar os dados num ficheiro para evitar fazer sempre isto todas as vezes que o programa corre
	}
	catch(int e)
	{
		 cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
	
	
	//trainSVM();
	
	//http://stackoverflow.com/questions/13689666/how-to-train-and-predict-using-bag-of-words
	//http://www.morethantechnical.com/2011/08/25/a-simple-object-classifier-with-bag-of-words-using-opencv-2-3-w-code/
	
	return 0; 
}

//Separar o treino e extracção de descriptors -> Gravar em ficheiros