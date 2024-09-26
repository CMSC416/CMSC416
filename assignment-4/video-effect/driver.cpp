#include <algorithm>// fill_n
#include <iostream> // cerr, cout
#include <string>   // string
#include <vector>   // vector
#include <fstream>  // ofstream
#include <cstdint>  // uint8_t
#include <cassert>  // assert
#include <opencv2/opencv.hpp>   // VideoCapture, VideoWriter, Mat, cvtColor
#include <cuda_runtime_api.h> // cudaMalloc, cudaFree, cudaMemcpy

/*
 *  Driver program for reading and writing the videos. You shouldn't need to edit this.
 */

/* set the maximum number of cuda streams */
constexpr int MAX_CUDA_STREAMS = 8;

namespace VideoUtilities {
/*  
 *  Utility class which wraps OpenCV for video IO.
 */
class VideoUtility {
    public:
        VideoUtility(std::string const& inputFilename, std::string const& outputFilename) 
            : inVideo_(inputFilename) {
            /* setup input video and get properties */
            if (!inVideo_.isOpened()) {
                std::cerr << "Unable to open video file: " << inputFilename << std::endl;
                std::exit(1);
            }
            width_ = inVideo_.get(cv::CAP_PROP_FRAME_WIDTH);
            height_ = inVideo_.get(cv::CAP_PROP_FRAME_HEIGHT);
            numFrames_ = inVideo_.get(cv::CAP_PROP_FRAME_COUNT);
            inVideo_.set(cv::CAP_PROP_CONVERT_RGB, 1);

            /* setup output video writer */
            int otherCodec = cv::VideoWriter::fourcc('m','p','4','v');
            double inputFPS = inVideo_.get(cv::CAP_PROP_FPS);
            outVideo_.open(outputFilename, otherCodec, inputFPS, cv::Size(width_, height_), true);
            outVideo_.set(cv::VIDEOWRITER_PROP_QUALITY, 100);
            if (!outVideo_.isOpened()) {
                std::cerr << "Unable to open video file: " << outputFilename << std::endl;
                std::exit(1);
            }

            /* allocated scratch frame */
            scratch_ = new float[width_*height_*3];
        }

        ~VideoUtility() { delete[] scratch_; }

        /* recommended batch size if we want to limit mb being offloaded to gpu */
        int getBatchSize(int mbLimit=1000) {
            int frameSize = width_ * height_* 3 * sizeof(float) / 1000000;
            return mbLimit/frameSize + (mbLimit%frameSize==0 ? 0 : 1);
        }

        /* loads next batch into batch. returns true if there are still more frames */
        bool nextFrames(std::vector<float *> &batch) {

            for (int i = 0; i < batch.size(); i += 1) {
                bool success = inVideo_.read(tmpReadMatUnsigned_);
                if (!success) {
                    /* clear remaining elements in rawData */
                    for (int j = batch.size()-1; j >= i; j -= 1) {
                        cudaFree(batch.at(j));
                        batch.pop_back();
                    }
                    break;
                }

                /* this driver is not set up to handle odd video formats */
                assert(tmpReadMatUnsigned_.type() == CV_8UC3 && tmpReadMatUnsigned_.isContinuous());
                tmpReadMatUnsigned_.convertTo(tmpReadMatFloat_, CV_32FC3, 1/255.0f);

                /* copy to gpu */ 
                cudaMemcpy(batch.at(i), tmpReadMatFloat_.data, width_*height_*3*sizeof(float), cudaMemcpyHostToDevice);
            }

            return !batch.empty();
        }

        /* write a batch of frames into video file */
        void writeFrames(std::vector<float *> &batch) {
            for (auto ptr : batch) {
                cudaMemcpy(scratch_, ptr, width_*height_*3*sizeof(float), cudaMemcpyDeviceToHost);
                cv::Mat tmpMat(height_, width_, CV_32FC3, scratch_);
                tmpMat.convertTo(tmpMat, CV_8UC3, 255.0f);
                outVideo_.write(tmpMat);
            }
        }

        /* dump csv of frame */
        void dumpFrame(std::vector<float *> &batch, int idx, std::string const& fname) {
            cudaMemcpy(scratch_, batch.at(idx), width_*height_*3*sizeof(float), cudaMemcpyDeviceToHost);

            std::ofstream outFile(fname);
            for (int i = 0; i < height_; i += 1) {
                for (int j = 0; j < width_; j += 1) {
                    int flatIdx = i*width_*3 + j*3;
                    outFile << static_cast<unsigned>(static_cast<uint8_t>(scratch_[flatIdx+0]*255.0f)) << ","
                            << static_cast<unsigned>(static_cast<uint8_t>(scratch_[flatIdx+1]*255.0f)) << ","
                            << static_cast<unsigned>(static_cast<uint8_t>(scratch_[flatIdx+2]*255.0f));

                    if (j != width_ - 1)
                        outFile << ",";
                }
                outFile << "\n";
            }
            outFile.close();
        }

        void allocateFrames(std::vector<float *> &rawData, int batchSize=10) {
            rawData.resize(batchSize);
            for (auto &ptr : rawData) {
                cudaMalloc((void **)&ptr, width_*height_*3*sizeof(float));
            }
        }

        void freeFrames(std::vector<float *> &rawData) {
            for (auto ptr : rawData)
                cudaFree(ptr);

            rawData.clear();
        }

        int width() const { return width_; }
        int height() const { return height_; }
        int numFrames() const { return numFrames_; }

    private:
        cv::VideoCapture inVideo_;
        cv::VideoWriter outVideo_;
        cv::Mat tmpReadMatUnsigned_, tmpReadMatFloat_;
        float *scratch_;
        int width_, height_, numFrames_;
};
}   // namespace VideoUtilities

/* Kernels defined here. Feel free to add if you want. */
float identityKernel[9] = {0.0f,0.0f,0.0f,
                           0.0f,1.0f,0.0f,
                           0.0f,0.0f,0.0f};
float edgeKernel[9] = {-1.0f,-1.0f,-1.0f,
                       -1.0f, 8.0f,-1.0f,
                       -1.0f,-1.0f,-1.0f};
float sharpenKernel[9] = { 0.0f,-1.0f, 0.0f,
                          -1.0f, 5.0f,-1.0f,
                           0.0f,-1.0f, 0.0f};
float blurKernel[25] = {0.004f,0.016f,0.023f,0.016f,0.004f,
                        0.016f,0.064f,0.094f,0.064f,0.016f,
                        0.023f,0.094f,0.141f,0.094f,0.023f,
                        0.016f,0.064f,0.094f,0.064f,0.016f,
                        0.004f,0.016f,0.023f,0.016f,0.004f};

float *getKernel(std::string const& kernelName, int &kernelWidth, int &kernelHeight) {
    float *cpuKernel, *gpuKernel;
    if (kernelName == "identity") {
        kernelWidth = 3;
        kernelHeight = 3;
        cpuKernel = identityKernel;
    } else if (kernelName == "edge") {
        kernelWidth = 3;
        kernelHeight = 3;
        cpuKernel = edgeKernel;
    } else if (kernelName == "sharpen") {
        kernelWidth = 3;
        kernelHeight = 3;
        cpuKernel = sharpenKernel;
    } else if (kernelName == "blur") {
        kernelWidth = 5;
        kernelHeight = 5;
        cpuKernel = blurKernel;
    } else {
        std::cerr << "Unknown kernel: " << kernelName << std::endl;
        std::exit(1);
    }

    /* move kernel onto gpu */
    cudaMalloc((void **)&gpuKernel, kernelWidth*kernelHeight*sizeof(float));
    cudaMemcpy(gpuKernel, cpuKernel, kernelWidth*kernelHeight*sizeof(float), cudaMemcpyHostToDevice);
    return gpuKernel;
}

void createStreams(cudaStream_t streams[MAX_CUDA_STREAMS]) {
    for (int i = 0; i < MAX_CUDA_STREAMS; i += 1) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
}

void destroyStreams(cudaStream_t streams[MAX_CUDA_STREAMS]) {
    for (int i = 0; i < MAX_CUDA_STREAMS; i += 1) {
        cudaStreamDestroy(streams[i]);
    }
}

/* external function for doing convolutions on a batch */
extern float convolveFrames(std::vector<float *> const& framesIn, std::vector<float *> &framesOut, int width, int height, 
                            float const* kernel, int kernelWidth, int kernelHeight,
                            cudaStream_t *streams, int numStreams,
                            int gridSizeX, int gridSizeY);


/*
 *  Driver program for reading, applying effects to, and writing out video files.
 */
int main(int argc, char **argv) {

    if (argc != 8 && argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <kernel_name> <grid_size_x> <grid_size_y> <?frame_idx> <?frame_file>" << std::endl;
        std::exit(1);
    }

    const std::string inputFilename(argv[1]);
    const std::string outputFilename (argv[2]);
    const std::string kernelName(argv[3]);
    const int gridSizeX = std::stoi(std::string(argv[4]));
    const int gridSizeY = std::stoi(std::string(argv[5]));
    int frameIdx = -1;
    std::string frameFilename;
    if (argc > 6) {
        frameIdx = std::stoi(std::string(argv[6]));
        frameFilename = std::string(argv[7]);
    }

    int kernelWidth, kernelHeight;
    float *kernel = getKernel(kernelName, kernelWidth, kernelHeight);

    cudaStream_t streams[MAX_CUDA_STREAMS];
    createStreams(streams);

    VideoUtilities::VideoUtility video(inputFilename, outputFilename);

    /* create frame batches */
    std::vector<float *> inputBatch, outputBatch;
    const int batchSize = video.getBatchSize(1500);
    video.allocateFrames(inputBatch, batchSize);
    video.allocateFrames(outputBatch, batchSize);

    /* do effects and write out */
    int batchCounter = 0;
    float duration = 0.0f;
    while (video.nextFrames(inputBatch)) {
        duration += convolveFrames(inputBatch, outputBatch, video.width(), video.height(), kernel, kernelWidth, 
            kernelHeight, streams, MAX_CUDA_STREAMS, gridSizeX, gridSizeY);
        video.writeFrames(outputBatch);

        if (video.numFrames() / frameIdx == batchCounter) {
            video.dumpFrame(outputBatch, frameIdx % batchSize, frameFilename);
        }
        batchCounter += 1;
    }

    /* clean up */
    video.freeFrames(inputBatch);
    video.freeFrames(outputBatch);
    cudaFree(kernel);
    destroyStreams(streams);

    /* write out timing info */
    float framesProcessedPerSecond = (batchCounter*batchSize) / duration;
    std::cout << "Total time: " << duration << " s\n";
    std::cout << "Frames processed per second: " << framesProcessedPerSecond << " frames/s\n";
}