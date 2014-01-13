#include <iostream>
#include <stdexcept>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

void medianBlurElab(cv::Mat_<unsigned char>& img, int k){
	cv::Mat_<unsigned char> temp;
	cv::medianBlur(img,temp,k);
	img = temp.clone();
}

/************* Main *****************/
int main(int argc, char *argv[]){
	try{
		if(argc<3){
			std::cout << "Wrong number of parameter";
			return 1;
		}
		
		cv::Mat_<unsigned char> src = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat_<unsigned char> elabImage = src.clone();

		bool medianBlur = true;
		int medianBlurKernelSize = 3; 

		if(medianBlur)
			medianBlurElab(elabImage,medianBlurKernelSize);

		cv::imwrite(argv[2], elabImage);
	}catch(...){
		return 2;
	}
	return 0;
}