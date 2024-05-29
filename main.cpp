
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <exception>

const char* kernelSource = R"(
__kernel void vector_add(__global const float* A, __global const float* B, __global float* C, int n) {
    int id = get_global_id(0);
    if (id < n) {
        C[id] = A[id] + B[id];
    }
}

__kernel void max(__global const float *arr, __global const int* shape, int dims, __global const int *axis, int numAxis) {

}

__kernel void max1d(__global const float *arr, )
__kernel void maxOverAxis(__global const float *arr, __global const int *shape, int dims, int axis) {

}
)";

int main() {
    // Initialize data
    const int N = 1056;
    std::vector<float> A(N, 1.0f); // Vector A with all elements 1.0
    std::vector<float> B(N, 2.0f); // Vector B with all elements 2.0
    std::vector<float> C(N, 0.0f); // Result vector C

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
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, A.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, B.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);

        // Create and build the program
        cl::Program::Sources sources;
        sources.push_back({kernelSource, strlen(kernelSource)});
        cl::Program program(context, sources);
        program.build(devices);

        // Create the kernel
        cl::Kernel kernel(program, "vector_add");

        // Set the kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, N);

        // Execute the kernel
        cl::NDRange global(N);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

        // Read back the result from the device to host
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * N, C.data());

        // Print the results
        for (int i = 1024; i < 1056; ++i) { // Print first 10 elements for verification
            std::cout << C[i] << " ";
        }
        std::cout << std::endl;

    } catch (cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        return -1;
    }

    return 0;
}
