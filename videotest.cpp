#include <opencv2/video/video.hpp>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
/* 
This code needs 3 arguments: the relative path to the 
video file being analyzed, the Low Threshold being used in
the Canny Filter, and the ratio between the Low and High 
Thresholds. It is currently set up to do both RGB Canny
filtering and Hue Canny filtering, although that is easily
changed. Adjust the waitkey(time) for a longer delay between
frames to make it run slower or faster.
*/
using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
	int lowThresh,ratio;
	stringstream conv;
	if(argc<4){cout<<"Not enough arguments"<<endl;return -1;}
	string source = argv[1];
	conv<<argv[2]<<endl<<argv[3];
	conv>>lowThresh>>ratio;
	//Loading the video into the opencv VideoCapture object
	VideoCapture cptsrc(source);
	if(!cptsrc.isOpened()){cout<<"Could not open "<<source; return -1;}
	Mat HSV,col,gray;
	Mat hue,dstc,dsth;
	Mat top,bot;
	vector<Mat> hsv;
	while(true)
	{
		//Loading a single frame into the Matrix object
		cptsrc>>col;
		//The only way the code knows the video is over is if the frame is empty
		if(col.empty()){cout<<"That's all folks!"<<endl;break;}
		//Filling the gray container with the grayscale of the RGB frame
		cvtColor(col,gray,CV_BGR2GRAY);
		//Filling the HSV container with the HSV version of the RGB frame
		cvtColor(col,HSV,CV_BGR2HSV);
		//Splitting the HSV frame into its three parts, in that order.
		split(HSV,hsv);
		//Gaussian smoothing on all of the hsv components
		for (int i = 0; i < hsv.size(); ++i)
		{
			blur(hsv[i],hsv[i],Size(3,3));
		}
		//The hue spectrum is periodic, so the brightest and the darkest are very close in value
		hsv[0] = hsv[0]+25;
		blur(gray,gray,Size(3,3));
		Canny(hsv[0],hue,lowThresh,lowThresh*ratio,3,true);
		//If you want to see saturation, uncomment the next line and add a window to view it.
		// Canny(hsv[1],sat,lowThresh,lowThresh*ratio,3,true);
		Canny(gray,gray,lowThresh,lowThresh*ratio,3,true);
		//filling destination color and destionation hue with black
		dstc = Scalar::all(0);
		dsth = Scalar::all(0);
		//copying the pixels defined as edges in gray, from color, to the all black container
		col.copyTo(dstc,gray);
		hsv[0].copyTo(dsth,hue);
		//combining two images side by side
		hconcat(col,dstc,top);
		hconcat(hsv[0],dsth,bot);
		//displaying
		namedWindow("Source,RGB",WINDOW_AUTOSIZE);
		namedWindow("Hue,Saturation Canny",WINDOW_AUTOSIZE);
		imshow("Source,RGB",top);
		imshow("Hue,Saturation Canny",bot);
		//delay time
		waitKey(30);
	}
}