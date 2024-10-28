CXX = g++
CXXFLAGS = -std=c++11 -O2
OPENCV_ROOT=/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/opencv-4.5.2-xxyodykxk3vuw64tlvm6sujgaxnctgep
CUDA_ROOT=/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/cuda-11.6.2-eonihhhvlh4s2d6riyb7al2qivzn477u
CXX_INCLUDES = -I$(OPENCV_ROOT)/include/opencv4 -I$(CUDA_ROOT)/include
CXX_LIBS = -L$(OPENCV_ROOT)/lib64 -L$(CUDA_ROOT)/lib64 -lopencv_core -lopencv_videoio -lopencv_imgproc -lcudart

CUDA = nvcc
CUDAFLAGS = -std=c++11 -O2 -Xcompiler "$(CXXFLAGS)"
GENCODE = --generate-code arch=compute_80,code=sm_80

TARGET = video-effect

all: $(TARGET)

video-effect.o: video-effect.cu
	$(CUDA) $(CUDAFLAGS) $(GENCODE) -o $@ -c $<

video-effect: driver.cpp video-effect.o
	$(CXX) $(CXXFLAGS) $(CXX_INCLUDES) -o $@ $^ $(CXX_LIBS)

clean:
	rm $(TARGET) *.o
