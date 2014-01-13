#include <iostream>
#include <stdexcept>
#include <algorithm> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

void medianBlurElab(cv::Mat_<unsigned char>& img, int k){
	cv::Mat_<unsigned char> temp;
	cv::medianBlur(img,temp,k);
	img = temp.clone();
	cv::imwrite("0_medianBlur.png", img);
}
void computeSquaredColorWeight(const cv::Mat_<unsigned char>& img, std::vector<float>& w){
	w = std::vector<float>(img.cols,0);
	for(int j = 0; j < img.rows; ++j)
		for(int i = 0; i < img.cols; ++i){
			unsigned int c = static_cast<unsigned char>(img.at<unsigned char>(j,i));
			unsigned int c2 = c*c;
			w[i] += c2;
		}

	for(int i = 0; i < img.cols; ++i)
		w[i] /= img.rows;

	const float maxVal = *(std::max_element(w.begin(),w.end()));
	const float minVal = *(std::min_element(w.begin(),w.end()));
	for(int i = 0; i < img.cols; ++i)
		w[i] = (w[i]-minVal)/(maxVal-minVal);
}

void applyColumnMap(cv::Mat_<unsigned char>& img, const std::vector<float>& w, float saturationFactor){
	for(int j = 0; j < img.rows; ++j)
		for(int i = 0; i < img.cols; ++i){
			const float scale = (0.15+w[i])*saturationFactor;

			img.at<unsigned char>(j,i) = std::min(scale*img.at<unsigned char>(j,i),255.f);
		}
}

void logColumnMap(const std::vector<float>& w, int rows, const std::string& name){
	cv::Mat_<float> log(rows,w.size());
	for(int j = 0; j < rows; ++j)
		for(int i = 0; i < w.size(); ++i){
			log.at<float>(j,i) = w[i]*255;
		}
	cv::imwrite(name, log);
}

void bilateralFilterOnMap(std::vector<float>& w, int radius, float radSigma, float wSigma){
	std::vector<float>& c_w = w;
	for(int i = radius; i < w.size()-radius; ++i){
		float tW = 0;
		float val = 0;
		for(int r = -radius; r <= radius; ++r){
			float rw = std::exp(-(r*r)/(2*radSigma*radSigma));
			float ww = std::exp(-std::pow(w[i]-w[i+r],2)/(2*wSigma*wSigma));
			float ew = rw*ww;
			val += c_w[i+r]*ew;
			tW += ew;
		}
		w[i] = val/tW;
	}
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
		float saturationFactor = 2.f;

		if(medianBlur)
			medianBlurElab(elabImage,medianBlurKernelSize);

		std::vector<float> squaredColorWeight;
		computeSquaredColorWeight(elabImage,squaredColorWeight);
		logColumnMap(squaredColorWeight,elabImage.rows,"1_squaredColorWeights.png");

		bilateralFilterOnMap(squaredColorWeight,5,2,0.05);
		logColumnMap(squaredColorWeight,elabImage.rows,"2_bilateralWeights.png");

		applyColumnMap(elabImage,squaredColorWeight,saturationFactor);

		cv::imwrite(argv[2], elabImage);
	}catch(...){
		return 2;
	}
	return 0;
}