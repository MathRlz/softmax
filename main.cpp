
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

vector<size_t> getDims(const vector<size_t> &oldDims, const vector<int> &axes) {
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

int main() {
    const vector<size_t> dims = {2, 3, 3};
    size_t size = accumulate(dims.begin(), dims.end(), 1, multiplies<size_t>());
    vector<float> array = { 1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                      
                        10, 11, 12,
                     13, 14, 15,
                     16, 17, 18};
    const vector<int> axes = {0, 1};
    const float alpha = 1.0f;

    cout.precision(8);
    cout << scientific;

    cout << "Array:\n";
    printArray(array, dims);

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

        // Allocate device memory and transfer input data to device
        cl::Buffer arrBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, array.data());
        cl::Buffer dimsBuffer(context, dims.begin(), dims.end(), true);
        // A bit too much memory consumption
        cl::Buffer reductionBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size);
        cl::Buffer reductionDimBuffer(context, CL_MEM_READ_WRITE, sizeof(size_t) * dims.size());
        cl::Buffer dimCntrBuffer(context, dimCntr.begin(), dimCntr.end(), true);
        cl::Buffer valDimsCntrBuffer(context, valDimsCntr.begin(), valDimsCntr.end(), true);
        int numDims = static_cast<int>(dims.size());


        // Create and build the program
        cl::Program::Sources sources;
        sources.push_back(readFile("softmax.cl"));
        cl::Program program(context, sources);
        program.build(devices);

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

    } catch (cl::Error& err) {
        cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
        return -1;
    }

    return 0;
}
