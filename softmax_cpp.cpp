#include <iostream>
#include <vector>
#include <limits>
#include <utility>

using namespace std;

// Function to find the maximum value along each axis
vector<float> max(const vector<float> &arr, const vector<size_t> &dims, int axis) {
    axis = dims.size() - 1 - axis;
    size_t stride = 1;
    for (int i = 0; i < axis; i++) {
        stride *= dims[i];
    }

    size_t size = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        if (static_cast<int>(i) != axis) {
            size *= dims[i];
        }
    }

    // When I hit my axis dimension (stride) then move enough to change to the next object
    // startPos = floor(i / stride) * dimSizesUpToMe;
    size_t dimSizesUpTo = stride * dims[axis];

    vector<float> result(size, 0);
    for (size_t i = 0; i < size; i++) {

        auto max = numeric_limits<float>::min();
        size_t startPos = (i / stride) * dimSizesUpTo + i % stride;

        cout << i << " | ";
        for (size_t j = 0; j < dims[axis]; j++) {
            size_t pos = startPos + j * stride;
            cout << pos << " ";
            float val = *reinterpret_cast<const float*>(arr.data() + pos);
            if (val > max) {
                max = val;
            }
        }
        cout << endl;
        result[i] = max;
    }
    return result;
}

void printArray(const vector<float> &arr, const vector<size_t> &dims) {
    vector<size_t> dimCntrs;
    size_t dimCntr=1;
    for (size_t i = 0; i < dims.size(); i++) {
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

        for (size_t j = 0; j < dimCntrs.size(); j++) {
            if (mods[j] == dimCntrs[j] - 1) {
                cout << "]" << endl;
            }
        }

        cout << ", ";
    }
}

int main() {
    // Example 3-dimensional array (replace with your own array)
    vector<float> arr = { 1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                      
                        10, 11, 12,
                     13, 14, 15,
                     16, 17, 18,
                     
                    19, 20, 21,
                    22, 23, 24,
                    25, 26, 27};
    vector<size_t> dims = {3, 3, 3};
    printArray(arr, dims);


    // Find the maximum value along each axis
    int axis = 0;
    vector<size_t> newDims;
    for (size_t i = 0; i < dims.size(); i++) {
        if (static_cast<int>(i) == axis) continue;
        newDims.push_back(dims[i]);
    }
    printArray(max(arr, dims, 0), newDims);
    printArray(max(arr, dims, 1), newDims);
    printArray(max(arr, dims, 2), newDims);


    // Output the maximum value along each axis

    return 0;
}
