
#include <iostream>
#include <stdexcept>

/************* Main *****************/
int main(int argc, char *argv[]){
	try{
		if(argc<3){
			std::cout << "Wrong number of parameter";
			return 1;
		}
		
	}catch(...){
		return 2;
	}
	return 0;
}