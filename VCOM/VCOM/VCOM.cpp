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

#define NUM_WORDS 100
#define SVM_LIMIT 10000 


string SVM_FILE = "SVM";
string DATA_DIRECTORY = "C:\\Dataset\\";

string convertInt(int number)
{
	stringstream ss;//create a stringstream
	ss << number;//add number to the stream
	return ss.str();//return a string with the contents of the stream
}

bool openImage(const string &f, Mat &image, int mode)
{
	string filename= DATA_DIRECTORY + "cars\\"+f+".image.png";
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

bool openMask(const string &f, Mat &image, int mode, int maskNumber)
{
	string filename=DATA_DIRECTORY + "cars\\"+f+".mask."+ convertInt(maskNumber) + ".png";
	if (mode==1)
		image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	else
		image=imread(filename);
	if( !image.data)
		return false;

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

Vector<string> images;
Mat image;
vector<Mat> imageMasks;
Mat tmpImage;

vector<KeyPoint> keypointsAll;
//we can have multiple cars on the picture
vector<vector<KeyPoint>> keypointsCar;
vector<KeyPoint> keypointsNotCar;

Mat descriptors;
Mat dictionary;

Ptr<DescriptorExtractor > descriptorExtractor;
Ptr<FeatureDetector> featureDetector;
Ptr<DescriptorMatcher> descriptorMatcher;
Mat training_descriptors;

Mat response_hist;

// Set up SVM's parameters
CvSVMParams SVM_Params;

Mat samples;
Mat labels(0,1,CV_32FC1);
Mat class_label;

CvSVM SVMClassifier;


int main( int argc, char** argv ) 
{
	try
	{
		SVM_Params.svm_type    = CvSVM::C_SVC;
		SVM_Params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, SVM_LIMIT, FLT_EPSILON );
		//(flags: finish when reaches last or good value, limit of iterations, rate)


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

		//open the file with the training images
		ifstream infile(DATA_DIRECTORY + "cars_train.txt");
		string line;
		int i=0;

		while(getline(infile,line))
		{
			//if(i>0)
			//break;
			images.push_back(line);

			//try to read image with the corresponding filename. We want to read in grayscale so the descriptors are color invariant.
			openImage(line,image,1);

			featureDetector->detect(image,keypointsAll);
			descriptorExtractor->compute(image,keypointsAll,descriptors);    
			training_descriptors.push_back(descriptors);

			if(i==0)
			{
				drawKeypoints(image, keypointsAll, tmpImage);
				imshow("Original",tmpImage);
			}
			cout<<"Reading Image: "<<line<<"\nImage number: "<<i<<endl;

			i++;
		}
		infile.close();

		cout<<"All Training Images Read"<<endl;
		cout<<"Total descriptors: "<<training_descriptors.rows<<endl;

		BOWKMeansTrainer bowTrainer(NUM_WORDS, TermCriteria(), 1, KMEANS_PP_CENTERS);
		BOWImgDescriptorExtractor bowExtractor(featureDetector, descriptorMatcher);

		cout<<"Generating Dictionary"<<endl;
		bowTrainer.add(training_descriptors);
		dictionary = bowTrainer.cluster();
		bowExtractor.setVocabulary(dictionary);

		cout<<"Waiting for Key"<<endl;
		waitKey(0);
		cout<<"Training SVM"<<endl;


		//verificar se existem dados em memória e ler do ficheiro
		string svmfilename =SVM_FILE + "f" + convertInt(detectorInput) + "m" + convertInt(matcherInput) + "w" + convertInt(NUM_WORDS) + ".xml";
		if(fileExists(svmfilename))
		{
			SVMClassifier.load(svmfilename.c_str());
			cout<<"SVM file loaded"<<endl;
		}
		else
		{
			//Aqui começa a segunda parte do treino, treinar a SVM
			samples = Mat(0,dictionary.rows,response_hist.type());
			for(int i = 0 ; i < images.size() ; i++)
			{
				cout<<"Image: "<<images[i]<<endl;

				keypointsCar.clear();
				keypointsNotCar.clear();

				//try to read image with the corresponding filename. We want to read in grayscale so the descriptors are color invariant.
				openImage(images[i],image,1);
				imageMasks.clear();
				int maskNumber = 0;
				for(bool exists_masks = true; exists_masks;)
				{
					exists_masks = openMask(images[i], tmpImage, 1, maskNumber);
					if(exists_masks)
					{
						imageMasks.push_back(tmpImage.clone());
						keypointsCar.push_back(vector<KeyPoint>());
						maskNumber++;
					}
				}

				cout<<"Number of masks: "<<maskNumber<<endl;
				featureDetector->detect(image,keypointsAll);

				for (int j = 0 ; j < keypointsAll.size() ; j++)
				{

					bool has_mask = false;
					for (int k = 0 ; k < maskNumber && !has_mask; k++)
					{
						//add to keypoints that are car
						if((uchar)imageMasks[k].at<uchar>((int)keypointsAll[j].pt.y, (int)keypointsAll[j].pt.x) == 76)
						{
							//cout<<"Position: "<<(int)keypointsAll[j].pt.y <<" "<<(int)keypointsAll[j].pt.x<<endl;
							//cout<<"Mask: "<<(int)imageMasks[k].at<uchar>((int)keypointsAll[j].pt.y, (int)keypointsAll[j].pt.x)<<endl;
							has_mask=true;
							keypointsCar[k].push_back(keypointsAll[j]);
						}
					}

					//not a single mask has detected a car
					if(!has_mask)
						//add to keypoints that aren't car
							keypointsNotCar.push_back(keypointsAll[j]);
				}
				if(i==0)
				{
					drawKeypoints(image, keypointsAll, tmpImage);
					imshow("All Keypoints",tmpImage);
					drawKeypoints(image, keypointsCar[0], tmpImage);
					imshow("Car Keypoints",tmpImage);

					drawKeypoints(image, keypointsNotCar, tmpImage);
					imshow("Outside Car Keypoints",tmpImage);
				}

				for (int k = 0 ; k < maskNumber ; k++)
				{
					descriptorExtractor->compute(image,keypointsCar[k],descriptors);
					bowExtractor.compute(image, keypointsCar[k], response_hist);
					class_label = Mat::ones(response_hist.rows, 1, CV_32FC1);
					labels.push_back(class_label);
					samples.push_back(response_hist);
				}


				descriptorExtractor->compute(image,keypointsNotCar,descriptors);
				bowExtractor.compute(image, keypointsNotCar, response_hist);
				class_label = Mat::zeros(response_hist.rows, 1, CV_32FC1);
				labels.push_back(class_label);
				samples.push_back(response_hist);
			}

			cout<<"Waiting for Key";
			waitKey(0);
			cout<<"Training SMV";

			Mat samples_32f;
			samples.convertTo(samples_32f, CV_32F);
			SVMClassifier.train(samples_32f,labels, Mat(), Mat(), SVM_Params);
			SVMClassifier.save(svmfilename.c_str());

			Mat result;
			bowExtractor.compute(image, keypointsAll,result);
		}

		//Testing SMV
		//open the file with the testing images
		infile = ifstream(DATA_DIRECTORY + "cars_test.txt");
		i=0;

		while(getline(infile,line))
		{
			images.push_back(line);

			//try to read image with the corresponding filename. We want to read in grayscale so the descriptors are color invariant.
			openImage(line,image,1);

			//TODO compute window

			bowExtractor.compute(image, keypointsAll, response_hist);

			float res = SVMClassifier.predict(response_hist);
			cout<<"Resposta: "<<res<<endl;
		}
		infile.close();

		cout<<"All Test Images Read"<<endl;


	}
	catch(Exception e)
	{
		cout << "An exception occurred. Exception Nr. " << e.msg << '\n';
	}

	//http://stackoverflow.com/questions/13689666/how-to-train-and-predict-using-bag-of-words
	//http://www.morethantechnical.com/2011/08/25/a-simple-object-classifier-with-bag-of-words-using-opencv-2-3-w-code/

	return 0; 
}

//Separar o treino e extracção de descriptors -> Gravar em ficheiros