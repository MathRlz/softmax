#include <iostream>
#include <vector>
#include <limits>
#include <utility>
#include <cmath>

using namespace std;

// Function to find the maximum value along each axis
vector<float> max(const vector<float> &arr, const vector<size_t> &dims, int axis) {
    size_t stride = 1;
    for (int i = dims.size() - 1; i > axis; i--) {
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

        for (size_t j = 0; j < dims[axis]; j++) {
            size_t pos = startPos + j * stride;
            float val = *reinterpret_cast<const float*>(arr.data() + pos);
            if (val > max) {
                max = val;
            }
        }
        result[i] = max;
    }
    return result;
}

// Find sum along an axis
vector<float> sum(const vector<float> &arr, const vector<size_t> &dims, int axis) {
    size_t stride = 1;
    for (int i = dims.size() - 1; i > axis; i--) {
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

        float sum = 0;
        size_t startPos = (i / stride) * dimSizesUpTo + i % stride;

        for (size_t j = 0; j < dims[axis]; j++) {
            size_t pos = startPos + j * stride;
            float val = *reinterpret_cast<const float*>(arr.data() + pos);
            sum += val;
        }
        result[i] = sum;
    }
    return result;
}

void sub(vector<float> &arr, const vector<size_t> &dims, const vector<float> &vals, int axis) {
    size_t stride = 1;
    for (int i = dims.size() - 1; i > axis; i--) {
        stride *= dims[i];
    }

    size_t size = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        if (static_cast<int>(i) != axis) {
            size *= dims[i];
        }
    }

    size_t dimSizesUpTo = stride * dims[axis];

    for (size_t i = 0; i < size; i++) {
        size_t startPos = (i / stride) * dimSizesUpTo + i % stride;

        for (size_t j = 0; j < dims[axis]; j++) {
            size_t pos = startPos + j * stride;
            float &val = *reinterpret_cast<float*>(arr.data() + pos);
            val -= vals[i];
        }
    }
}

void div(vector<float> &arr, const vector<size_t> &dims, const vector<float> &vals, int axis) {
    size_t stride = 1;
    for (int i = dims.size() - 1; i > axis; i--) {
        stride *= dims[i];
    }

    size_t size = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        if (static_cast<int>(i) != axis) {
            size *= dims[i];
        }
    }

    size_t dimSizesUpTo = stride * dims[axis];

    for (size_t i = 0; i < size; i++) {
        size_t startPos = (i / stride) * dimSizesUpTo + i % stride;

        for (size_t j = 0; j < dims[axis]; j++) {
            size_t pos = startPos + j * stride;
            float &val = *reinterpret_cast<float*>(arr.data() + pos);
            val /= vals[i];
        }
    }
}

void exp(vector<float> &arr) {
    for(size_t i = 0; i < arr.size(); i++) {
        arr[i] = exp(arr[i]);
    }
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

vector<size_t> getDims(const vector<size_t> &oldDims, int axis) {
    vector<size_t> newDims;
    for (size_t i = 0; i < oldDims.size(); i++) {
        if (static_cast<int>(i) == axis) continue;
        newDims.push_back(oldDims[i]);
    }
    return newDims;
}

int main() {
    // Example 3-dimensional array (replace with your own array)
    vector<float> arr = { 1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                      
                        10, 11, 12,
                     13, 14, 15,
                     16, 17, 18};
    vector<size_t> dims = {2, 3, 3};
    cout.precision(8);
    cout << scientific;
    cout << "Array:\n";
    printArray(arr, dims);


    int axis = 0;
    cout << "Max " << axis << ":\n";
    // Find the maximum value along each axis
    auto maxVals = max(arr, dims, axis);
    auto reducedDims = getDims(dims, axis);
    printArray(maxVals, reducedDims);

    cout << "Array after Substraction:\n";
    sub(arr, dims, maxVals, axis);
    printArray(arr, dims);

    cout << "Array after Exponentiation:\n";
    exp(arr);
    printArray(arr, dims);

    cout << "AX_sum:\n";
    auto sumVals = sum(arr, dims, axis);
    printArray(sumVals, reducedDims);

    cout << "Softmax:\n";
    div(arr, dims, sumVals, axis);
    printArray(arr, dims);


    // Output the maximum value along each axis

    return 0;
}
