/*
Credit to Chen Jia and Nghia Ho for the awesome trajectory
and mapping structures for video stabilization. The Kalman 
Filter notation is purposefully modelled after the great
paper by Greg Welch and Gary Bishop, posted online at:
http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
Modifications by Elad Michael
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
	// "+"
	friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory &rx){
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x,y,a);
	}

    double x;
    double y;
    double a; // angle
};

vector<Trajectory> K_Filter(Trajectory,Trajectory,Trajectory,vector<Trajectory>);

int main( int argc, char** argv ) //arguments will be ./executable VideoFileName FrameRadius
{
	const int horiz_crop=20;
	bool Phone=false;

	if(argc!=2){cout<<"Incorrect Number of Arguments"<<endl;return -1;}

	VideoCapture cptsrc(argv[1]);
	if(!cptsrc.isOpened()){cout<<"Could not open file"<<endl; return -1;}

	Mat cur,prev,cur_grey,prev_grey;

	cptsrc>>prev;
	if(prev.cols > 700) {resize(prev, prev, Size(prev.cols/2, prev.rows/2));Phone=true;}
	cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

	Mat T(2,3,CV_64F);

	Mat last_T;
	double a,x,y; a=x=y=0; //accumulated x,y, and angular flow

	Trajectory X;//posteriori state estimate
	Trajectory	X_;//priori estimate
	Trajectory P;// posteriori estimate error covariance
	Trajectory P_;// priori estimate error covariance
	Trajectory z;//actual measurement
	double pstd = 4e-03;//can be changed
	double mstd = 0.25;//can be changed
	Trajectory Q(pstd,pstd,pstd);// process noise covariance
	Trajectory R(mstd,mstd,mstd);// measurement noise covariance 
	Trajectory H(1,1,1);//weighting matrix from measurement to true trajectory
	Trajectory A(1,1,1);//Guess from previous state to next state

	int vert_border = horiz_crop*prev.rows/prev.cols; // get the aspect ratio correct
	int k=1; //frame count

	vector<Trajectory>temp,learn;
	learn.push_back(Q);
	learn.push_back(R);
	learn.push_back(H);
	learn.push_back(A);

	while(true) 
	{
		cptsrc >> cur;
		if(cur.data == NULL) {cout<<"That's all folks!"<<endl;return 1;}
		if(cur.cols > 700) {resize(cur, cur, Size(cur.cols/2, cur.rows/2));}

		cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

		vector <Point2f> prev_corner, cur_corner; //vectors for points of interest mappings
		vector <Point2f> prev_corner_g, cur_corner_g; //vectors for GOOD points of interest
		vector <uchar> status; //status of each point of interest
		vector <float> err;	//error of each point of interest

		//input,output,max number of poi, quality ratio to best poi,min distance between poi
		goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30); 
		//try to find poi from last image in next image
		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

		//Only take good matches
		for(size_t i=0; i < status.size(); i++) 
		{
			if(status[i]) 
			{
				prev_corner_g.push_back(prev_corner[i]);
				cur_corner_g.push_back(cur_corner[i]);
			}
		}

		T = estimateRigidTransform(prev_corner_g,cur_corner_g,false);

		// in rare cases no transform is found. We'll just use the last known good transform.
		if(T.data == NULL) {last_T.copyTo(T);}
		T.copyTo(last_T);

		//retrieve data from the Transformation matrix:
		double dx = T.at<double>(0,2);
		double dy = T.at<double>(1,2);
		double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

		//update accumulated mapping parameters
		x+=dx;
		y+=dy;
		a+=da;

		z = Trajectory(x,y,a);

		if(k==1)
		{
			X = Trajectory(0,0,0); //First guess, no transformation
			P = Trajectory(1,1,1); //Initialize error variance to identity
		}
		else
		{
			temp = K_Filter(X,P,z,learn);
			X = temp[0];
			P = temp[1];
			temp.clear();
		}
		double diff_x = X.x-x;
		double diff_y = X.y-y;
		double diff_a = X.a-a;

		dx += X.x-x; //error in prediction along with calculated flow
		dy += X.y-y; //same...
		da += X.a-a; //same...

		T.at<double>(0,0) = cos(da); //2-D Rotation Matrix
		T.at<double>(0,1) = -sin(da); //with the smoothed
		T.at<double>(1,0) = sin(da); //angles calculated
		T.at<double>(1,1) = cos(da);

		T.at<double>(0,2) = dx;
		T.at<double>(1,2) = dy;

		Mat curs; //current_smoothed

		warpAffine(prev, curs, T, cur.size());

		curs = curs(Range(vert_border, curs.rows-vert_border), Range(horiz_crop, curs.cols-horiz_crop));
		// Resize curs back to cur size, for better side by side comparison
		resize(curs, curs, cur.size());

		cur = cur(Range(vert_border, cur.rows-vert_border), Range(horiz_crop, cur.cols-horiz_crop));

		resize(cur, cur, curs.size());


		Mat diff = Mat(curs.rows,curs.cols,curs.type());
		cv::absdiff(prev,curs,diff);
		Scalar errs = cv::sum(diff);
		double error = (errs[0]+errs[1]+errs[2])/(diff.rows*diff.cols*255);
		cout<<"Frame "<<k<<" Error: "<<error<<endl;
		if(Phone)
		{
			Mat rot1,rot2;
			transpose(curs,rot1);
			transpose(diff,rot2);
			flip(rot1,rot1,1);
			flip(rot2,rot2,1);
			// Now draw the original and stablised side by side for coolness
			Mat canvas = Mat::zeros(rot1.rows, rot1.cols*2+10, rot1.type());
			// prev.copyTo(canvas(Range::all(), Range(0, curs.cols)));		
			rot1.copyTo(canvas(Range::all(), Range(0, rot1.cols)));
			rot2.copyTo(canvas(Range::all(), Range(rot2.cols+10, rot2.cols*2+10)));

			//if video is larger than the screen, scale it DOWN
			if(canvas.cols > 1366) {resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));}

			cout<<canvas.cols<<endl;
			imshow("before and after", canvas);	
			waitKey(1);		
		}
		else
		{
			// Now draw the original and stablised side by side for coolness
			Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());
			// prev.copyTo(canvas(Range::all(), Range(0, curs.cols)));
			diff.copyTo(canvas(Range::all(), Range(0, curs.cols)));		
			curs.copyTo(canvas(Range::all(), Range(curs.cols+10, curs.cols*2+10)));

			//if video is larger than the screen, scale it DOWN
			if(canvas.cols > 1366) {resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));}

			cout<<canvas.cols<<endl;
			imshow("before and after", canvas);
			waitKey(20);
		}	
		//
		prev = cur.clone();//cur.copyTo(prev);
		cur_grey.copyTo(prev_grey);

		k++;
	}
	return 0;
}

vector<Trajectory> K_Filter(Trajectory X,Trajectory P, Trajectory z,vector<Trajectory> learn)
{
	Trajectory Q = learn[0];
	Trajectory R = learn[1];
	Trajectory H = learn[2];
	Trajectory A = learn[3];
	//Time Updates
	Trajectory X_=A*X;
	Trajectory P_=A*P*A+Q;
	//Measure Updates
	Trajectory K=(H*P_)/(H*H*P_+R);
	X = X_+K*(z-X_);
	P=(Trajectory(1,1,1)-K*H)*P_;
	vector<Trajectory> Result;
	Result.push_back(X); Result.push_back(P);
	return Result;
}