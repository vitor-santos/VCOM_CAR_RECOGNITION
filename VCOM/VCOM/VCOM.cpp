#include "stdafx.h"
#include <vector>
#include <opencv/cv.h>
#include <fstream>
#include <iostream>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

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
		image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	else
		image=imread(filename);
	if( !image.data) {
		cout << " --(!) Error reading image " << filename << endl;
		return false;
	}
	return true;
}

bool openMask(const string &f, Mat &image, int mode)
{
	string filename="C:\\Dataset\\cars\\"+f+".mask.0.png";
	if (mode==1)
		image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	else
		image=imread(filename);
	if( !image.data) {
		cout << " --(!) Error reading image " << filename << endl;
		return false;
	}
	return true;
}



int main( int argc, char** argv ) 
{
	Vector<string> images;
	Mat image;
	Mat imageMask;

	vector<KeyPoint> keypointsAll;
	vector<KeyPoint> keypointsCar;
	vector<KeyPoint> keypointsNotCar;
	vector<vector<KeyPoint>> results;	
	Mat descriptors;
	Mat dictionary;

	Ptr<DescriptorExtractor > descriptorExtractor;
	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorMatcher> descriptorMatcher;
	Mat training_descriptors;

	Mat response_hist;

	 // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	Mat class_label;
	// Train the SVM with the car
	CvSVM SVM;

	
	//verficar se existem dados em memória e ler do ficheiro
	

	int detectorInput, matcherInput;
	cout<<"Insert the number corresponding to the desired FeatureDetector algorithm:"<<endl<<"1 - SIFT"<<endl<<"2 - SURF"<<endl; 
	cin>>detectorInput;
	cout<<"Insert the number corresponding to the desired DescriptorMatcher algorithm:"<<endl<<"1 - FlannBased"<<endl<<"2 - BruteForce"<<endl;
	cin>>matcherInput;	
	
	if(detectorInput==1)
	{
		descriptorExtractor= Ptr<DescriptorExtractor>(new SiftDescriptorExtractor());
		featureDetector = Ptr<FeatureDetector>(new SiftFeatureDetector());
	}
	else 
	{
		descriptorExtractor=Ptr<DescriptorExtractor>(new SurfDescriptorExtractor());
		featureDetector = Ptr<FeatureDetector>(new SurfFeatureDetector());
	}
	
	if(matcherInput==1)
		descriptorMatcher = Ptr<DescriptorMatcher>(new FlannBasedMatcher());
	else
		//TODO make brute force
		descriptorMatcher = Ptr<DescriptorMatcher>(new BFMatcher());

		
	
	training_descriptors = Mat(1,descriptorExtractor->descriptorSize(),descriptorExtractor->descriptorType());
	//Initial training and clustering
	try
	{
		//open the file with the training images
		ifstream infile("C:\\Dataset\\cars_test.txt");
		string line;
		int i=0;

		while(getline(infile,line))
		{
			if(i>2)
				break;
			images.push_back(line);
			
			//try to read image with the corresponding filename. We want to read in grayscale so the descriptors are color invariant.
			openImage(line,image,1);

			featureDetector->detect(image,keypointsAll);
			descriptorExtractor->compute(image,keypointsAll,descriptors);    
			training_descriptors.push_back(descriptors);
			cout<<"Reading Image: "<<line<<"\nImage number: "<<i<<endl;
			i++;
		}

		infile.close();
		

		cout<<"Total descriptors: "<<training_descriptors.rows<<endl;

		BOWKMeansTrainer bowTrainer(100, TermCriteria(), 1, KMEANS_PP_CENTERS);
		BOWImgDescriptorExtractor bowExtractor(featureDetector, descriptorMatcher);

		bowTrainer.add(training_descriptors);

		dictionary = bowTrainer.cluster();
		bowExtractor.setVocabulary(dictionary);


		//Aqui começa a segunda parte do treino, treinar a SVM
		for(int i = 0 ; i < images.size() ; i++)
		{
			cout<<"Image number: "<<i<<endl;

			//try to read image with the corresponding filename. We want to read in grayscale so the descriptors are color invariant.
			openImage(images[i],image,1);
			openMask(images[i],imageMask,1);

			featureDetector->detect(image,keypointsAll);

			keypointsCar.clear();
			keypointsNotCar.clear();
			
			for (int j = 0 ; j < keypointsAll.size() ; j++)
			{
				cout<<"Position: "<<(int)keypointsAll[j].pt.x <<" "<<(int)keypointsAll[j].pt.y<<endl;
				cout<<"Mask: "<<(int)imageMask.at<char>((int)keypointsAll[j].pt.x, (int)keypointsAll[j].pt.y)<<endl;
				//add to keypoints that are car
				if((int)imageMask.at<char>((int)keypointsAll[j].pt.x, (int)keypointsAll[j].pt.y) != 0)
					keypointsCar.push_back(keypointsAll[j]);
				//add to keypoints that aren't car
				else
					keypointsNotCar.push_back(keypointsAll[j]);
			}
			descriptorExtractor->compute(image,keypointsCar,descriptors);
			bowExtractor.compute(image, keypointsCar, response_hist);

			//one element to classify, one element defines class, class element is a CV_8U
			class_label = Mat(1, 1, CV_8U, 1);

			// Train the SVM with the car
			try
			{
				SVM.train(response_hist, class_label, Mat(), Mat(), params);
			}
			catch (Exception e)
			{
				cout<<e.msg;
			}
			descriptorExtractor->compute(image,keypointsNotCar,descriptors);
			bowExtractor.compute(image, keypointsNotCar, response_hist);

			//one element to classify, one element defines class, class element is a CV_8U
			class_label = Mat(1, 1, CV_8U, 1);

			// Train the SVM not with the car
			SVM.train(response_hist, class_label, Mat(), Mat(), params);

		}


		
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