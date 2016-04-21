#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat src, src_gray,disp;
Mat dst, detected_edges;

//This code is basically verbatim from the opencv tutorials
//I personally hate using global variables if I can avoid it
//a step by step analysis is done on the website for this code

int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";
static void CannyThreshold(int, void*)
{
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    dst = Scalar::all(0);
    src.copyTo( dst, detected_edges);
    hconcat(src,dst,disp);
    imshow( window_name, disp);
}
int main( int, char** argv )
{
  src = imread( argv[1]);
  if( src.empty() )
    { return -1; }
  Size size(360,640);//the dst image size,e.g.100x100
  resize(src,src,size);//resize imag
  dst.create( src.size(), src.type() );
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  namedWindow( window_name, WINDOW_AUTOSIZE );
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
  CannyThreshold(0, 0);
  waitKey(0);
  return 0;
}