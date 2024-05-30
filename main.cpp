
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
    // Initialize data
    const int N = 1056;
    
    const vector<size_t> dims = {2, 3, 3};
    size_t size = accumulate(dims.begin(), dims.end(), 1, multiplies<size_t>());

    vector<float> array = { 1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                      
                        10, 11, 12,
                     13, 14, 15,
                     16, 17, 18};

    //srand(unsigned(time(nullptr)));
    //generate(array.begin(), array.end(), rand);

    cout.precision(8);
    cout << scientific;

    cout << "Array:\n";
    printArray(array, dims);
    vector<float> resultArray(size, 0.0f);

    const vector<int> axes = {0};
    const auto reducedDims = getDims(dims, axes);
    size_t reducedSize = accumulate(reducedDims.begin(), reducedDims.end(), 1, multiplies<size_t>());

    auto dimCntr = getDimCntr(dims);
    auto valDimsCntr = getDimCntr(reducedDims);

    
    try {
        // Get all platforms (drivers)
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Get a list of devices on this platform
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Create a context for the devices
        cl::Context context(devices);

        // Create a command queue for the first device
        cl::CommandQueue queue(context, devices[0]);

        // Allocate device memory and transfer input data to device
        cl::Buffer arrBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, array.data());
        // A bit too much memory consumption
        cl::Buffer reductionBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size);
        cl::Buffer dimsBuffer(context, dims.begin(), dims.end(), true);
        cl::Buffer reducedDimBuffer(context, reducedDims.begin(), reducedDims.end(), true);
        cl::Buffer dimCntrBuffer(context, dimCntr.begin(), dimCntr.end(), true);
        cl::Buffer valDimsCntrBuffer(context, valDimsCntr.begin(), valDimsCntr.end(), true);
        int numDims = static_cast<int>(dims.size());


        // Create and build the program
        cl::Program::Sources sources;
        sources.push_back(readFile("softmax.cl"));
        cl::Program program(context, sources);
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
                for (auto& device : devices) {
                    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                    std::cerr << "Build log for device " << device.getInfo<CL_DEVICE_NAME>() << ":\n";
                    std::cerr << buildLog << "\n";
                }
            }
            return -1;
        }

        // Create the kernel
        cl::Kernel maxKernel(program, "max");
        // Set the kernel arguments
        maxKernel.setArg(0, arrBuffer);
        maxKernel.setArg(1, dimsBuffer);
        maxKernel.setArg(2, reductionBuffer);
        maxKernel.setArg(3, numDims);
        
        maxKernel.setArg(4, axes[0]);
        cl::NDRange globalReduce(reducedSize);
        queue.enqueueNDRangeKernel(maxKernel, cl::NullRange, globalReduce, cl::NullRange);
        
        vector<float> maxValues(reducedSize);
        queue.enqueueReadBuffer(reductionBuffer, CL_TRUE, 0, sizeof(float) * reducedSize, maxValues.data());
        cout << "Max:\n";
        printArray(maxValues, reducedDims);


        cl::Kernel subAndExpKernel(program, "sub_and_exp");
        subAndExpKernel.setArg(0, arrBuffer);
        subAndExpKernel.setArg(1, dimsBuffer);
        subAndExpKernel.setArg(2, reductionBuffer);
        subAndExpKernel.setArg(3, reducedDimBuffer);
        subAndExpKernel.setArg(4, dimCntrBuffer);
        subAndExpKernel.setArg(5, valDimsCntrBuffer);
        subAndExpKernel.setArg(6, numDims);

        cl::NDRange global(size);
        queue.enqueueNDRangeKernel(subAndExpKernel, cl::NullRange, global, cl::NullRange);
        vector<float> subAndExpVal(size);
        queue.enqueueReadBuffer(arrBuffer, CL_TRUE, 0, sizeof(float) * size, subAndExpVal.data());
        cout << "Sub and exp\n";
        printArray(subAndExpVal, dims);

        cl::Kernel sumKernel(program, "sum");
        sumKernel.setArg(0, arrBuffer);
        sumKernel.setArg(1, dimsBuffer);
        sumKernel.setArg(2, reductionBuffer);
        sumKernel.setArg(3, numDims);
        sumKernel.setArg(4, axes[0]);
        queue.enqueueNDRangeKernel(sumKernel, cl::NullRange, globalReduce, cl::NullRange);

        cl::Kernel divKernel(program, "div");
        divKernel.setArg(0, arrBuffer);
        divKernel.setArg(1, dimsBuffer);
        divKernel.setArg(2, reductionBuffer);
        divKernel.setArg(3, reducedDimBuffer);
        divKernel.setArg(4, dimCntrBuffer);
        divKernel.setArg(5, valDimsCntrBuffer);
        divKernel.setArg(6, numDims);
        queue.enqueueNDRangeKernel(divKernel, cl::NullRange, global, cl::NullRange);


        // Execute the kernel

        // Read back the result from the device to host
        queue.enqueueReadBuffer(arrBuffer, CL_TRUE, 0, sizeof(float) * size, array.data());
        cout << "Softmax:\n";
        printArray(array, dims);

        std::cout << std::endl;

    } catch (cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        return -1;
    }

    return 0;
}
