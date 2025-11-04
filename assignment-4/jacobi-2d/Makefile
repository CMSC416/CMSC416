CXX = g++
CXXFLAGS = -std=c++11
CUDA = nvcc
CUDAFLAGS = -std=c++11 -O2 -Xcompiler "$(CXXFLAGS)"
GENCODE=--generate-code arch=compute_80,code=sm_80

TARGET = jacobi-2d

.PHONE: all debug
all: $(TARGET)

debug: CXXFLAGS += -g -fsanitize=address
debug: all

$(TARGET): $(TARGET).cu
	$(CUDA) $(CUDAFLAGS) $(GENCODE) -o $@ $<

clean:
	rm $(TARGET)
