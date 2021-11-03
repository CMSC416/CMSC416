CXX = g++
CXXFLAGS = -std=c++11 -O2
OPENCV_ROOT=/software/spack-software/2020.05.14/linux-rhel8-ivybridge/gcc-8.4.0/opencv-4.2.0-f7wcimpv42jmxqe22uuf5ghmsza6hvic
CUDA_ROOT=/software/spack-software/2020.05.14/linux-rhel8-ivybridge/gcc-8.4.0/cuda-10.2.89-av2y5i5crqaikkyl7smufeb7jkuwcgju
CXX_INCLUDES = -I$(OPENCV_ROOT)/include/opencv4 -I$(CUDA_ROOT)/include
CXX_LIBS = -L$(OPENCV_ROOT)/lib64 -lopencv_core -lopencv_videoio -lopencv_imgproc -lcudart

CUDA = nvcc
CUDAFLAGS = -std=c++11 -O2 -Xcompiler "$(CXXFLAGS)"
GENCODE = --generate-code arch=compute_35,code=sm_35

TARGET = video-effect

all: $(TARGET)

video-effect.o: video-effect.cu
	$(CUDA) $(CUDAFLAGS) $(GENCODE) -o $@ -c $<

video-effect: driver.cc video-effect.o
	$(CXX) $(CXXFLAGS) $(CXX_INCLUDES) -o $@ $^ $(CXX_LIBS)

clean:
	rm $(TARGET) *.o
