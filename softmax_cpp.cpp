#include <iostream>
#include <vector>
#include <limits>
#include <utility>
#include <cmath>
#include <functional>
#include <algorithm>

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

void sub2(vector<float> &arr, const vector<size_t> &dims, const vector<float> &vals, const std::vector<int> &axes) {
    vector<size_t> dimCntrs(dims.size(), 0);
    size_t dimCntr = 1;
    for (int i = dims.size()-1; i >= 0; i--) {
        dimCntrs[i] = dimCntr;
        dimCntr *= dims[i];
        cout << i << " " << dimCntrs[i] << endl;
    }

    vector<size_t> valsDims(dims.size());
    cout << "ValDims\n";
    for (size_t i = 0; i < valsDims.size(); i++) {
        if (find(axes.begin(), axes.end(), i) == end(axes)) {
            valsDims[i] = dims[i];
        } else {
            valsDims[i] = 1;
        }
        cout << valsDims[i] << " ";
    }
    cout <<endl;

    vector<size_t> valsDimCntr(dims.size(), 0);
    dimCntr = 1;
    for (int i = valsDims.size() - 1; i >= 0; i--) {
        valsDimCntr[i] = dimCntr;
        dimCntr *= valsDims[i];
    }


    std::vector<size_t> pos(dims.size(), 0);
    for (size_t i = 0; i < arr.size(); i++) {

        pos[pos.size()-1] = i % dims[dims.size()-1];
        for (size_t j = 0; j < pos.size() - 1; j++) {
            pos[j] = (i / dimCntrs[j]) % dims[j];
        }

        cout << "Pos( ";
        for (size_t j = 0; j < pos.size(); j++) {
            cout << pos[j] << ", ";
        }
        cout << ")\n";

        size_t valPos = 0;
        for (size_t j = 0; j < pos.size(); j++) {
            if (valsDims[j] != 1) {
               valPos += pos[j] * valsDimCntr[j];
            }
        }
            

        cout << "Val pos = " << valPos << endl;

        float &val = *reinterpret_cast<float*>(arr.data() + i);
        val -= vals[valPos];

    }
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

void softmax(vector<float> &arr, const vector<size_t> &dims, const vector<int> &unsortedAxes) {
    cout << "Array:\n";
    printArray(arr, dims);
    auto sortedAxes = unsortedAxes;
    std::sort(sortedAxes.begin(), sortedAxes.end(), less<size_t>());
    int axis = sortedAxes[0];
    cout << "Going over axis " << axis;
    auto reducedDims = getDims(dims, axis);
    // Find the maximum value along each axis
    auto maxVals = max(arr, dims, axis);

    cout << "Max :\n";
    printArray(maxVals, reducedDims);

    sub2(arr, dims, maxVals, {axis});
    cout << "Array after Substraction:\n";
    printArray(arr, dims);

    exp(arr);
    cout << "Array after Exponentiation:\n";
    printArray(arr, dims);

    auto sumVals = sum(arr, dims, axis);
    cout << "AX_sum:\n";
    printArray(sumVals, reducedDims);

    div(arr, dims, sumVals, axis);
    cout << "Softmax:\n";
    printArray(arr, dims);
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

    softmax(arr, dims, {0, 1});

    cout << "FINITO\n";
    printArray(arr, dims);


    // Output the maximum value along each axis

    return 0;
}
