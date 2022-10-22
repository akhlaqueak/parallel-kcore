__device__ bool binarySearch(unsigned int* arr, unsigned int end, unsigned int val){
    unsigned int begin = 0;
    unsigned int mid = (begin+end)/2;
    while(begin<=end){
        if(val == arr[mid]) return true;
        else if(val<arr[mid]) end = mid-1;
        else begin = mid+1;
    }
    return false;
}
// counts occurance of v in data[st:en]
// __device__ int countAtWarp(unsigned int* data, unsigned int st, unsigned int en, unsigned int v){
//     bool pred;
//     unsigned int laneid = LANEID;
//     unsigned int sum = 0;
//     for(; st<en; st+=32){
//         k = st+laneid;
//         pred = false;
//         // not applying binary search here, as it may diverge the warp
//         if(k < en)
//             pred = (v == data[k]);
//         sum += __popc(__ballot_sync(FULLl, pred));
//     }
//     return sum;
// }
__device__ int searchAny(unsigned int* data, unsigned int st, unsigned int en, unsigned int v){
    bool pred;
    unsigned int laneid = LANEID;
    unsigned int res;
    for(unsigned int k; st<en; st+=32){
        k = st+laneid;
        pred = false;
        // not applying binary search here, as it may diverge the warp
        if(k < en)
            pred = (v == data[k]);
        res = __ballot_sync(FULL, pred);
        if(res!=0) return true;
    }
    return false;
}

__device__ int searchAnyPX(char* data, unsigned int st, unsigned int en){
    bool pred;
    unsigned int laneid = LANEID;
    unsigned int res;
    for(unsigned int k; st<en; st+=32){
        k = st+laneid;
        pred = false;
        if(k < en)
            pred = (data[k]==P || data[k]==X);
        res = __ballot_sync(FULL, pred);
        if(res!=0) return true;
    }
    return false;
}

__device__ int searchAnyP(char* data, unsigned int st, unsigned int en){
    bool pred;
    unsigned int laneid = LANEID;
    unsigned int res;
    for(unsigned int k; st<en; st+=32){
        k = st+laneid;
        pred = false;
        if(k < en)
            pred = (data[k]==P);
        res = __ballot_sync(FULL, pred);
        if(res!=0) return true;
    }
    return false;
}