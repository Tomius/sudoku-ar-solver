ALL:
	clang++ -std=c++11 *.cpp -Wall `pkg-config --cflags --libs opencv` -ltesseract
