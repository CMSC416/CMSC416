all: serial

CXX = g++
CXXFLAGS = -std=c++11 -O2

serial: serial.cpp
	$(CXX) $(CXXFLAGS) -o life $<

clean:
	rm -f serial.o life
