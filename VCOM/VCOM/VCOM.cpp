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
string VOCAB_FILE = "VOCAB";
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

float windowProportion = 3.0;

int truePositives = 0;
int undetectedPixels = 0;
int numberDetectedPixels = 0;
int maskPixels = 0;

float sucess = 0.0;
float maskCorrect = 0.0;
float maskMissed = 0.0;



int main( int argc, char** argv ) 
{
	try
	{
		string line;
		int i=0;

		SVM_Params.svm_type    = CvSVM::C_SVC;
		SVM_Params.C           = 0.1;
		SVM_Params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, SVM_LIMIT, FLT_EPSILON );
		//(flags: finish when reaches last or good value, limit of iterations, rate)

		

		cv::initModule_nonfree();


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

		BOWKMeansTrainer bowTrainer(NUM_WORDS, TermCriteria(), 1, KMEANS_PP_CENTERS);
		BOWImgDescriptorExtractor bowExtractor(featureDetector, descriptorMatcher);
			
		string vocabfilename =VOCAB_FILE + "f" + convertInt(detectorInput) + "m" + convertInt(matcherInput) + "w" + convertInt(NUM_WORDS) + ".xml";
		FileStorage vocabStorage;
		Mat vocabulary;
		if(fileExists(vocabfilename))
		{
			cout<<"Reading vocabulary from file"<<endl;
			vocabStorage.open(vocabfilename,FileStorage::READ);
			vocabStorage["Vocabulary"]>>dictionary;
			vocabStorage.release();

			//open the file with the training images
			ifstream infile(DATA_DIRECTORY + "cars_train.txt");
			while(getline(infile,line))
				images.push_back(line);
		}
		else
		{
			//open the file with the training images
			ifstream infile(DATA_DIRECTORY + "cars_train.txt");

			cout<<"Reading Training Images"<<endl;
			while(getline(infile,line))
			{
				if(i%10 == 0)
					cout<<"Reading Image: "<<line<<"\nImage number: "<<i<<endl;

				//if(i>0)
				//break;
				images.push_back(line);

				//try to read image with the corresponding filename. We want to read in grayscale so the descriptors are color invariant.
				openImage(line,image,1);

				featureDetector->detect(image,keypointsAll);
				descriptorExtractor->compute(image,keypointsAll,descriptors);    
				training_descriptors.push_back(descriptors);

				/*if(i==0)
				{
					drawKeypoints(image, keypointsAll, tmpImage);
					imshow("Original",tmpImage);
					cout<<"Waiting for Key"<<endl;
					waitKey(0);
					cout<<"Continuing"<<endl;
				}*/
				i++;
			}
			infile.close();

			cout<<"All Training Images Read"<<endl;
			cout<<"Total descriptors: "<<training_descriptors.rows<<endl;
			cout<<"Generating Dictionary and BoW Extractor"<<endl;
			bowTrainer.add(training_descriptors);
			dictionary = bowTrainer.cluster();

			vocabStorage.open(vocabfilename,FileStorage::WRITE);
			vocabStorage<<"Vocabulary"<<dictionary;
			vocabStorage.release();
		}
		bowExtractor.setVocabulary(dictionary);
		cout<<"Sucess"<<endl;
		//cout<<"Waiting for Key"<<endl;
		//waitKey(0);


		//verificar se existem dados em memória e ler do ficheiro
		string svmfilename =SVM_FILE + "f" + convertInt(detectorInput) + "m" + convertInt(matcherInput) + "w" + convertInt(NUM_WORDS) + ".xml";
		if(fileExists(svmfilename))
		{
			SVMClassifier.load(svmfilename.c_str());
			cout<<"SVM file loaded"<<endl;
		}
		else
		{
			cout<<"Reading Masks"<<endl;
	
			//Aqui começa a segunda parte do treino, treinar a SVM
			samples = Mat(0,dictionary.rows,response_hist.type());
			for(int i = 0 ; i < images.size() ; i++)
			{

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

				//inserting instances of car to SVM
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
				
				if(i==0)
				{
					drawKeypoints(image, keypointsAll, tmpImage);
					imshow("All Keypoints",tmpImage);
					drawKeypoints(image, keypointsCar[0], tmpImage);
					imshow("Car Keypoints",tmpImage);
					drawKeypoints(image, keypointsNotCar, tmpImage);
					imshow("Outside Car Keypoints",tmpImage);
					cout<<"Waiting for Key"<<endl;
					//waitKey(0);
					cout<<"Continuing"<<endl;
				}
				if(i%10 == 0)
				{
					cout<<"Reading Image: "<<line<<"\nImage number: "<<i<<endl;
					cout<<"Number of masks: "<<maskNumber<<endl;
				}
			}

			Mat samples_32f;
			samples.convertTo(samples_32f, CV_32F);
			SVMClassifier.train_auto(samples_32f,labels, Mat(), Mat(), SVM_Params);
			SVMClassifier.save(svmfilename.c_str());
		}

		cout<<"SVM obtained"<<endl;
		//cout<<"Waiting for Key"<<endl;
		//waitKey(0);
		cout<<"Testing SVM"<<endl;


		//Testing SMV
		//open the file with the testing images
		ifstream infile(DATA_DIRECTORY + "cars_test.txt");
		int numberTestImages = 0;
		while(getline(infile,line))
		{
			//try to read image with the corresponding filename. We want to read in grayscale so the descriptors are color invariant.
			openImage(line,image,1);
			numberTestImages++;

			imshow("All Image", image);

			Mat mask = Mat::zeros(image.rows, image.cols, CV_8U);
			tmpImage = Mat::zeros(image.rows, image.cols, image.type());

			for(int i = 0 ; i < (windowProportion * 2) - 1; i++)
			{
				for(int j = 0 ; j < (windowProportion * 2) -1 ; j++)
				{
					Mat dst_roi = image(Rect(i*image.cols*(1.0/(windowProportion*2.0)), j*image.rows*(1.0/(windowProportion*2.0)), (1.0/windowProportion)*image.cols, (1.0/windowProportion)*image.rows));
					imshow("Patch", dst_roi);
					//waitKey(0);
					featureDetector->detect(dst_roi,keypointsAll);
					if(keypointsAll.size() !=0)
					{
						bowExtractor.compute(dst_roi, keypointsAll, response_hist);
						float res = SVMClassifier.predict(response_hist);
						//cout<<"PREDICT: "<<res<<" "<<Rect(i*image.cols*(1.0/(windowProportion*2.0)), j*image.rows*(1.0/(windowProportion*2.0)), (1.0/windowProportion)*image.cols, (1.0/windowProportion)*image.rows)<<endl;
					
						if(res==1)
							for(int x = 0 ; x<(1.0/windowProportion)*image.cols ; x++)
								for(int y = 0 ; y <(1.0/windowProportion)*image.rows; y++)
									mask.at<uchar>(j*image.rows*(1.0/(windowProportion*2.0)) + y, i*image.cols*(1.0/(windowProportion*2.0)) + x) = 1;
					}
				}
			}
			image.copyTo(tmpImage, mask);

			//getting subwindows and testing there
			cout<<"Image: " + line<<endl;
			imshow("Current Prediction",tmpImage);
			//cout<<"Waiting for Key"<<endl;
			//waitKey(0);
			string savefilename= DATA_DIRECTORY + "results\\" + line + "_f" + convertInt(detectorInput) + "m" + convertInt(matcherInput) + "w" + convertInt(NUM_WORDS) + "d" + convertInt(windowProportion) + ".png";
			imwrite(savefilename , tmpImage);

			//counting the number of detected pixels on the mask
			for(int x = 0 ; x < tmpImage.rows ; x++)
				for(int y = 0 ; y < tmpImage.cols ; y++)
					if(mask.at<uchar>(x,y) !=0)
						numberDetectedPixels++;

			int maskNumber = 0;
			for(bool exists_masks = true; exists_masks;)
			{
				exists_masks = openMask(line, tmpImage, 1, maskNumber);
				if(exists_masks)
				{
					maskNumber++;
					for(int x = 0 ; x < tmpImage.rows ; x++)
						for(int y = 0 ; y < tmpImage.cols ; y++)
						{
							if (tmpImage.at<uchar>(x,y) ==76)
								maskPixels++;
							if (tmpImage.at<uchar>(x,y) ==76 && mask.at<uchar>(x,y) !=0)
							{
								//true positive
								truePositives++;
							}
							else if (tmpImage.at<uchar>(x,y) ==76 && mask.at<uchar>(x,y) ==0)
							{
								//undetected
								undetectedPixels++;
							}
						}
				}
			}
			if(numberDetectedPixels !=0)
				sucess += (truePositives*100.0)/(numberDetectedPixels);

			maskCorrect += (truePositives*100.0)/(maskPixels);
			maskMissed += (undetectedPixels*100.0)/(maskPixels);


			truePositives = 0;
			undetectedPixels = 0;
			numberDetectedPixels = 0;
			maskPixels = 0;
		}
		infile.close();

		cout<<"All Testing Images Read"<<endl;
		cout<<"Calculating Statistics"<<endl;

		string statisticsFilename= DATA_DIRECTORY + "results\\f" + convertInt(detectorInput) + "m" + convertInt(matcherInput) + "w" + convertInt(NUM_WORDS) + "d" + convertInt(windowProportion) + ".txt";
		ofstream myfile;
		myfile.open (statisticsFilename);
		myfile << "Sucess: " <<sucess/numberTestImages<<endl;
		myfile << "Percentage Mask Correct: " <<maskCorrect/numberTestImages<<endl;
		myfile << "Percentage Mask Missed: " <<maskMissed/numberTestImages<<endl;
			
		myfile.close();
	}
	catch(Exception e)
	{
		cout << "An exception occurred. Exception Nr. " << e.msg << '\n';
	}

	cout<<"Waiting for Key"<<endl;
	waitKey(0);

	//http://stackoverflow.com/questions/13689666/how-to-train-and-predict-using-bag-of-words
	//http://www.morethantechnical.com/2011/08/25/a-simple-object-classifier-with-bag-of-words-using-opencv-2-3-w-code/

	return 0; 
}