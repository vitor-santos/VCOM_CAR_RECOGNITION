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

#define SVM_LIMIT 10000
#define SVM_FILE "SVM.xml"

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

bool fileExists(string filename) {
	ifstream file(filename.c_str());
	if (file.is_open())
	{
		file.close();
		return true;
	}
	file.close();
	return false;
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
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, SVM_LIMIT, FLT_EPSILON );
	//(flags: finish when reaches last or good value, limit of iterations, rate)

	Mat samples;
	Mat labels(0,1,CV_32FC1);
	Mat class_label;

	CvSVM SVM;

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
		descriptorMatcher = Ptr<DescriptorMatcher>(new BFMatcher());

	training_descriptors = Mat(1,descriptorExtractor->descriptorSize(),descriptorExtractor->descriptorType());
	//verificar se existem dados em memória e ler do ficheiro
	if(fileExists(SVM_FILE))
		SVM.load(SVM_FILE);
	else
	{
		//Process of SVM training
		try
		{
			//open the file with the training images
			ifstream infile("C:\\Dataset\\cars_train.txt");
			string line;
			int i=0;

			while(getline(infile,line))
			{
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

			samples = Mat(0,dictionary.rows,response_hist.type());

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

				cout<<"ROWS: "<<imageMask.rows<<"COLUMNS: "<<imageMask.cols;

				for (int j = 0 ; j < keypointsAll.size() ; j++)
				{
					cout<<"Position: "<<(int)keypointsAll[j].pt.x <<" "<<(int)keypointsAll[j].pt.y<<endl;
					cout<<"Mask: "<<(uchar)imageMask.at<uchar>((int)keypointsAll[j].pt.x, (int)keypointsAll[j].pt.y)<<endl;
					//add to keypoints that are car
					if((uchar)imageMask.at<uchar>((int)keypointsAll[j].pt.x, (int)keypointsAll[j].pt.y) != 0)
						keypointsCar.push_back(keypointsAll[j]);
					//add to keypoints that aren't car
					else
						keypointsNotCar.push_back(keypointsAll[j]);
				}
				descriptorExtractor->compute(image,keypointsCar,descriptors);
				bowExtractor.compute(image, keypointsCar, response_hist);

				//cout<<"HIST: "<<response_hist.cols<<"\nVocabSize: "<<dictionary.dims<<"-"<<dictionary.rows<<"-"<<dictionary.cols;

				class_label = Mat::ones(response_hist.rows, 1, CV_32FC1);
				labels.push_back(class_label);
				samples.push_back(response_hist);

				descriptorExtractor->compute(image,keypointsNotCar,descriptors);
				bowExtractor.compute(image, keypointsNotCar, response_hist);

				class_label = Mat::zeros(response_hist.rows, 1, CV_32FC1);
				labels.push_back(class_label);
				samples.push_back(response_hist);
			}
			Mat samples_32f;
			samples.convertTo(samples_32f, CV_32F);
			SVM.train(samples_32f,labels, Mat(), Mat(), params);
			SVM.save(SVM_FILE);
		}
		catch(Exception e)
		{
			cout << "An exception occurred. Exception Nr. " << e.msg << '\n';
		}
	}

	//http://stackoverflow.com/questions/13689666/how-to-train-and-predict-using-bag-of-words
	//http://www.morethantechnical.com/2011/08/25/a-simple-object-classifier-with-bag-of-words-using-opencv-2-3-w-code/

	return 0; 
}

//Separar o treino e extracção de descriptors -> Gravar em ficheiros