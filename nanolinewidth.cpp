#include <iostream>
#include <stdexcept>
#include <algorithm> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

void medianBlurElab(cv::Mat_<unsigned char>& img, int k){
	cv::Mat_<unsigned char> temp;
	cv::medianBlur(img,temp,k*2+1);
	img = temp.clone();
	
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

void applyColumnMap(cv::Mat_<unsigned char>& img, const std::vector<float>& w, float saturationOffset, float saturationFactor){
	for(int j = 0; j < img.rows; ++j)
		for(int i = 0; i < img.cols; ++i){
			const float scale = (saturationOffset+w[i])*saturationFactor;
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
void paramList(){
	std::cout << "Single_nm_linewidth_measurement.exe input output (par value )*" << std::endl;
	std::cout << "Parameters:" << std::endl;
	std::cout << "medianBlurKernelRadius:\t-mr" << std::endl;
	std::cout << "bilateralRadius:\t-br" << std::endl;
	std::cout << "bilateralDistSigma:\t-bds" << std::endl;
	std::cout << "bilateralWeightSigma:\t-bws" << std::endl;
	std::cout << "saturationFactor:\t-sf" << std::endl;
	std::cout << "saturationOffset:\t-so" << std::endl;
}

/************* Main *****************/
int main(int argc, char *argv[]){
	try{
		if(argc<3){
			std::cout << "Wrong number of parameter";
			paramList();
			return 1;
		}
		int medianBlurKernelRadius	= 1; 
		int bilateralRadius			= 5; 
		float bilateralDistSigma	= 3;
		float bilateralWeightSigma	= 0.05;
		float saturationFactor		= 2.f;
		float saturationOffset		= 0.15f;

		for(int p = 3; p < argc; ++p){
			std::string param(argv[p]);
			if(param.compare("-mr") == 0 && p < argc-1){
				++p;
				medianBlurKernelRadius = atoi(argv[p]);
			}else if(param.compare("-br") == 0 && p < argc-1){
				++p;
				bilateralRadius = atoi(argv[p]);
			}else if(param.compare("-bds") == 0 && p < argc-1){
				++p;
				bilateralDistSigma = atof(argv[p]);
			}else if(param.compare("-bws") == 0 && p < argc-1){
				++p;
				bilateralWeightSigma = atof(argv[p]);
			}else if(param.compare("-sf") == 0 && p < argc-1){
				++p;
				saturationFactor = atof(argv[p]);
			}else if(param.compare("-so") == 0 && p < argc-1){
				++p;
				saturationOffset = atof(argv[p]);
			}else{
				std::cout << "Wrong parameter" << std::endl;
				paramList();
				return 3;
			}
		}

		cv::Mat_<unsigned char> src = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat_<unsigned char> elabImage = src.clone();

		if(medianBlurKernelRadius>0){
			medianBlurElab(elabImage,medianBlurKernelRadius);
			cv::imwrite("medianBlur.png", elabImage);
		}

		std::vector<float> squaredColorWeight;
		computeSquaredColorWeight(elabImage,squaredColorWeight);
		logColumnMap(squaredColorWeight,elabImage.rows,"squaredColorWeights.png");

		if(bilateralRadius > 0 && bilateralDistSigma > 0 && bilateralWeightSigma > 0){
			bilateralFilterOnMap(squaredColorWeight,bilateralRadius,bilateralDistSigma,bilateralWeightSigma);
			logColumnMap(squaredColorWeight,elabImage.rows,"bilateralWeights.png");
		}

		applyColumnMap(elabImage,squaredColorWeight,saturationOffset,saturationFactor);

		cv::imwrite(argv[2], elabImage);
	}catch(...){
		return 2;
	}
	return 0;
}