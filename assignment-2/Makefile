all: serial

CXX = g++
CXXFLAGS = -std=c++11 -O2

serial: serial.C
	$(CXX) $(CXXFLAGS) -o life $<

clean:
	rm -f serial.o life
