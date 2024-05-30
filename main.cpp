
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <exception>
#include <functional>
#include <algorithm>
#include <fstream>
#include <string>
#include <numeric>

using namespace std;

string readFile(const string &fileName) {
    ifstream file(fileName);
    return string(istreambuf_iterator<char>(file), {});
}

void printArray(const vector<float> &arr, const vector<size_t> &dims) {
    vector<size_t> dimCntrs;
    size_t dimCntr=1;
    for (int i = dims.size()-1; i >= 0; i--) {
        dimCntr *= dims[i];
        dimCntrs.push_back(dimCntr);
    }

    vector<size_t> mods(dims.size(), 0);
    for (size_t i = 0; i < arr.size(); i++) {
        for (size_t j = 0; j < dimCntrs.size(); j++) {
            mods[j] = i % dimCntrs[j];
            if (mods[j] == 0) {
                cout << "[";
            }
        }

        cout << arr[i];

        bool needEndl = false;
        for (size_t j = 0; j < dimCntrs.size(); j++) {
            if (mods[j] == dimCntrs[j] - 1) {
                cout << "]";
                needEndl = true;
            }
        }

        cout << ", ";
        if(needEndl) cout << endl;
    }
}

vector<size_t> getDims(const vector<size_t> &oldDims, const vector<uint> &axes) {
    vector<size_t> newDims(oldDims.size());
    for (size_t i = 0; i < oldDims.size(); i++) {
        if (find(axes.begin(), axes.end(), i) == end(axes)) {
            newDims[i] = oldDims[i];
        } else {
            newDims[i] = 1;
        }
    }
    return newDims;
}

vector<size_t> getDimCntr(const vector<size_t> &dims) {
    vector<size_t> dimCntr(dims.size());
    size_t tmp = 1;
    for (int i = dims.size()-1; i >= 0; i--) {
        dimCntr[i] = tmp;
        tmp *= dims[i];
    }
    return dimCntr;
}

float randFloat() {
    constexpr float LO = -1000.0f;
    constexpr float HI = 1000.0f;
    return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
}

uint32_t flp2 (uint32_t x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

uint32_t fnp2(uint32_t x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

bool isPow2(uint32_t x) {
    return (x & (x - 1)) == 0;
}

int main() {
    const vector<size_t> dims = {512, 2, 1};
    size_t size = accumulate(dims.begin(), dims.end(), 1, multiplies<size_t>());
    std::srand(unsigned(std::time(nullptr)));
    std::vector<float> array(size, 1.0f);

    //array[127] =123.4321f;
    //std::generate(array.begin(), array.end(), randFloat);
    /*vector<float> array = { 1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                      
                        10, 11, 12,
                     13, 14, 15,
                     16, 17, 18};
                     */
                     
    const vector<uint> axes = {0, 1};
    const float alpha = 1.0f;

    cout.precision(8);
    cout << scientific;

    cout << "Array:\n";
    //printArray(array, dims);

    auto dimCntr = getDimCntr(dims);
    auto valDimsCntr = getDimCntr(getDims(dims, axes));
    
    try {
        // Get all platforms (drivers)
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Get a list of devices on this platform
        vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Create a context for the devices
        cl::Context context(devices);

        // Create a command queue for the first device
        cl::CommandQueue queue(context, devices[0]);

        size_t maxWorkGroupSize;
        devices[0].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

        std::cout << "Max work group size: " << maxWorkGroupSize << std::endl;

        // Allocate device memory and transfer input data to device
        cl::Buffer arrBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, array.data());
        cl::Buffer reductionBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size);
        cl::Buffer reductionBuffer2(context, CL_MEM_READ_WRITE, sizeof(float) * size);

        cl::Buffer inDimsBuffer(context, dims.begin(), dims.end(), false);
        cl::Buffer outDimsBuffer(context, CL_MEM_READ_WRITE, sizeof(size_t) * dims.size());
        cl::Buffer outDims2Buffer(context, CL_MEM_READ_WRITE, sizeof(size_t) * dims.size());
        uint numDims = static_cast<uint>(dims.size());
        uint axis = axes[0];
        /*
        cl::Buffer dimsBuffer(context, dims.begin(), dims.end(), true);
        // A bit too much memory consumption
        cl::Buffer reductionBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size);
        cl::Buffer reductionDimBuffer(context, CL_MEM_READ_WRITE, sizeof(size_t) * dims.size());
        cl::Buffer dimCntrBuffer(context, dimCntr.begin(), dimCntr.end(), true);
        cl::Buffer valDimsCntrBuffer(context, valDimsCntr.begin(), valDimsCntr.end(), true);
        int numDims = static_cast<int>(dims.size());
        */


        // Create and build the program
        cl::Program::Sources sources;
        sources.push_back(readFile("softmax.cl"));
        cl::Program program(context, sources);
        program.build(devices);

        size_t localSize = 256;

        {
            size_t tmpSize = size;
            size_t desiredSize = size / dims[axis];
            size_t reducedDimSize = dims[axis];
            
            auto inBuffer = arrBuffer;
            auto inDims = inDimsBuffer;
            auto outBuffer = reductionBuffer;
            auto outDims = outDimsBuffer;

            cl::Kernel sumReduceKernel(program, "reduce_sum_ND");
            sumReduceKernel.setArg(6, numDims);
            sumReduceKernel.setArg(7, axis);

            while (reducedDimSize > 1) {
                size_t ndLocalSize = localSize;
                size_t ndGlobalSize = isPow2(tmpSize) ? tmpSize : fnp2(tmpSize);

                if (tmpSize < localSize) {
                    ndLocalSize = flp2(reducedDimSize);
                }
                
                reducedDimSize /= ndLocalSize;

                sumReduceKernel.setArg(0, static_cast<size_t>(localSize * sizeof(float)), nullptr);
                sumReduceKernel.setArg(1, inBuffer);
                sumReduceKernel.setArg(2, inDims);
                sumReduceKernel.setArg(3, outBuffer);
                sumReduceKernel.setArg(4, outDims);
                sumReduceKernel.setArg(5, static_cast<uint>(tmpSize));

                cout << "Size " << tmpSize << endl;
                cout << "GlobalSize " << ndGlobalSize << endl;
                cout << "LocalSize " << ndLocalSize << endl;
                queue.enqueueNDRangeKernel(sumReduceKernel, cl::NullRange, ndGlobalSize, ndLocalSize);

                
                vector<float> inputVec(tmpSize);
                queue.enqueueReadBuffer(inBuffer, CL_TRUE, 0, inputVec.size() * sizeof(float), inputVec.data());
                cout <<"Input ";
                for (auto val : inputVec) {
                    cout << val << " ";
                }
                cout <<endl;

                
                tmpSize = ndGlobalSize / ndLocalSize;

                vector<float> sumReduce(tmpSize);
                queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, sumReduce.size() * sizeof(float), sumReduce.data());
                for (auto val : sumReduce) {
                    cout << val << " ";
                }
                cout << endl;
                printArray(sumReduce, getDims(dims, {axis}));

                vector<size_t> outDimsVec(dims.size());
                queue.enqueueReadBuffer(outDims, CL_TRUE, 0, outDimsVec.size() * sizeof(size_t), outDimsVec.data());
                cout << "Dims: ";
                for (auto dim : outDimsVec) {
                    cout << dim << " ";
                }
                cout << endl;
                vector<size_t> inDimsVec(dims.size());
                queue.enqueueReadBuffer(inDims, CL_TRUE, 0, inDimsVec.size() * sizeof(size_t), inDimsVec.data());
                cout << "In Dims: ";
                for (auto dim : inDimsVec) {
                    cout << dim << " ";
                }
                cout << endl;
                
                inDims = outDims;
                outDims = (outDims == outDimsBuffer) ? outDims2Buffer : outDimsBuffer;
                inBuffer = outBuffer;
                outBuffer = (outBuffer == reductionBuffer) ? reductionBuffer2 : reductionBuffer;

            }
        }
        /*
        cl::Kernel reduceKernel(program, "reduce_sum_ND");
        reduceKernel.setArg(0, static_cast<size_t>(localSize * sizeof(float)), nullptr);
        reduceKernel.setArg(1, arrBuffer);
        reduceKernel.setArg(2, inDims);

        reduceKernel.setArg(2, reductionBuffer);
        reduceKernel.setArg(3, static_cast<unsigned int>(size));

y

        bool lastOp = false;
        while (tmpSize > 1) {
            size_t ndLocalSize = localSize;
            size_t ndGlobalSize = isPow2(tmpSize) ? tmpSize : fnp2(tmpSize);

            if (tmpSize < localSize) {
                ndLocalSize = flp2(tmpSize);
                ndGlobalSize = ndLocalSize * 2;
            }

            if (tmpSize == 2) {
                lastOp = true;
            }

            reduceKernel.setArg(0, static_cast<size_t>(ndLocalSize * sizeof(float)), nullptr);
            reduceKernel.setArg(1, bufferToReduce);
            reduceKernel.setArg(3, static_cast<unsigned int>(tmpSize));
            cout << "TMP size " << tmpSize << endl;
            cout << "Global work size " << ndGlobalSize << endl;
            cout << "Local work size " << ndLocalSize << endl;

            queue.enqueueNDRangeKernel(reduceKernel, cl::NullRange, cl::NDRange(ndGlobalSize), cl::NDRange(ndLocalSize));
            
            bufferToReduce = reductionBuffer;
            tmpSize = ndGlobalSize / ndLocalSize;

            if (lastOp) { tmpSize = 1; }

            vector<float> sumReduce(tmpSize);
            queue.enqueueReadBuffer(reductionBuffer, CL_TRUE, 0, sumReduce.size() * sizeof(float), sumReduce.data());
            for (auto val : sumReduce) {
                cout << val << " ";
            }
            cout << endl;

        }
        */

        /*

        { // mul
            cl::Kernel mulKernel(program, "mul");
            mulKernel.setArg(0, arrBuffer);
            mulKernel.setArg(1, alpha);
            queue.enqueueNDRangeKernel(mulKernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
        }

        { // max
            // Copy original dims
            queue.enqueueCopyBuffer(dimsBuffer, reductionDimBuffer, 0, 0, sizeof(size_t) * dims.size());

            cl::Kernel maxKernel(program, "max");
            maxKernel.setArg(1, reductionDimBuffer);
            maxKernel.setArg(2, reductionBuffer);
            maxKernel.setArg(3, numDims);

            auto tmpSize = size;
            auto tmpArrBuffer = arrBuffer;
            for (auto axis : axes) {
                tmpSize /= dims[axis];

                maxKernel.setArg(0, tmpArrBuffer);
                maxKernel.setArg(4, axis);
                tmpArrBuffer = reductionBuffer;

                queue.enqueueNDRangeKernel(maxKernel, cl::NullRange, cl::NDRange(tmpSize), cl::NullRange);
            }
        }

        { // sub and exp
            cl::Kernel subAndExpKernel(program, "sub_and_exp");
            subAndExpKernel.setArg(0, arrBuffer);
            subAndExpKernel.setArg(1, dimsBuffer);
            subAndExpKernel.setArg(2, reductionBuffer);
            subAndExpKernel.setArg(3, reductionDimBuffer);
            subAndExpKernel.setArg(4, dimCntrBuffer);
            subAndExpKernel.setArg(5, valDimsCntrBuffer);
            subAndExpKernel.setArg(6, numDims);

            queue.enqueueNDRangeKernel(subAndExpKernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
        }

        { // sum
            // Copy original dims
            queue.enqueueCopyBuffer(dimsBuffer, reductionDimBuffer, 0, 0, sizeof(size_t) * dims.size());

            cl::Kernel sumKernel(program, "sum");
            sumKernel.setArg(1, reductionDimBuffer);
            sumKernel.setArg(2, reductionBuffer);
            sumKernel.setArg(3, numDims);
            
            auto tmpArrBuffer = arrBuffer;
            auto tmpSize = size;
            for (auto axis : axes) {
                tmpSize /= dims[axis];
                
                sumKernel.setArg(0, tmpArrBuffer);
                sumKernel.setArg(4, axis);
                tmpArrBuffer = reductionBuffer;

                queue.enqueueNDRangeKernel(sumKernel, cl::NullRange, cl::NDRange(tmpSize), cl::NullRange);
            }
        }

        { // div
            cl::Kernel divKernel(program, "div");
            divKernel.setArg(0, arrBuffer);
            divKernel.setArg(1, dimsBuffer);
            divKernel.setArg(2, reductionBuffer);
            divKernel.setArg(3, reductionDimBuffer);
            divKernel.setArg(4, dimCntrBuffer);
            divKernel.setArg(5, valDimsCntrBuffer);
            divKernel.setArg(6, numDims);
            queue.enqueueNDRangeKernel(divKernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
        }

        // Read back the result from the device to host
        queue.enqueueReadBuffer(arrBuffer, CL_TRUE, 0, sizeof(float) * size, array.data());
        cout << "Softmax:\n";
        printArray(array, dims);
        cout << endl;
        */

    } catch (cl::Error& err) {
        cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
        return -1;
    }

    return 0;
}
