#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

using namespace std;

int main(){
    int *sum = new int[11];
    int *degrees= new int[10];
    for(int i=0;i<10;i++) degrees[i] = i+2;
    sum[0] = 0;
    partial_sum(degrees, degrees+10, sum+1);

    for(int i=0;i<11;i++) cout<<sum[i]<<" ";
    return 0;
}