
#include "../inc/buffer.h"

// __device__ inline unsigned int getWriteLoc(unsigned int* bufTail){
//     return atomicAdd(bufTail, 1);
// }
__device__ unsigned int ldg (const unsigned int * p)
{
    unsigned int out;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}


__device__ void writeToBuffer(Node* tail, unsigned int loc, unsigned int v){
    if(loc < tail->limit - BUFF_SIZE){ // write to prev node
        tail->prev->data[loc%BUFF_SIZE] = v;
    }else{    // write to current node
        tail->data[loc%BUFF_SIZE] = v;
    }
}


__device__ unsigned int readFromBuffer(Node* head, unsigned int loc){
    assert(head!=NULL); 
    unsigned int v = 55;
    if(loc < head->limit)
        v = head->data[loc%BUFF_SIZE];
    else
        v = head->next->data[loc%BUFF_SIZE];
    return v;
}



__device__ bool allocationRequired( Node* tail, unsigned int loc){
    if(tail==NULL)
        return (true); // first node is going to create.
    else
        return loc >= (tail->limit); //current limit exceed the requirement now
}

__device__ void allocateMemory(Node** tail, Node** head){
    Node* newNode = ( Node*) malloc(sizeof(Node));
    assert(newNode!=NULL);
    newNode->next = NULL;
    newNode->prev = NULL;
    if(tail[0]==NULL){ // in that case head is also NULL, this is the first node in linked list
        newNode -> limit = BUFF_SIZE; 
    }
    else{
        newNode -> limit = (tail[0]->limit+BUFF_SIZE);
        tail[0] -> next = newNode;
        newNode -> prev = tail[0];
    }

    tail[0] = newNode;

    if(head[0]==NULL) 
        head[0] = newNode;
    // printf("allocate... %d \n", newNode->limit);
}

__device__ void allocateMemoryMutex(Node** tail, Node** head, volatile unsigned int* lock){
    
    if(atomicExch((unsigned int*)lock, 1) == 0){        
        // printf("mutex %d %d\n", blockIdx.x, THID);
        allocateMemory(tail, head);
        lock[0] = 2; // not necessary to do it atomically, since it's the only thread in critical section
        __threadfence_block(); // it ensures the writes are realized to shared/global mem
    }
    while(lock[0]!=2);
}    
