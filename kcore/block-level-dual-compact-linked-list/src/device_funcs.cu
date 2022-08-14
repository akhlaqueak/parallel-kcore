
#include "../inc/device_funcs.h"

#include "../inc/buffer.h"
#include "../inc/scans.h"


__device__ void scanBlockHillis(unsigned int* addresses){
    // Hillis Steele Scan
    // todo check this code is working
    __syncthreads();
    int initVal = addresses[THID];

    for (unsigned int d = 1; d < BLK_DIM; d = d*2) {
        unsigned int newVal = addresses[THID];   
        if (int(THID - d) >= 0)  
            newVal += addresses[THID-d];  
        __syncthreads();
        addresses[THID] = newVal;
        __syncthreads();  
    }
        //Hillis-Steele Scan gives inclusive scan.
        //to get exclusive scan, subtract the initial values.
    addresses[THID] -= initVal;
    __syncthreads();  
}

__device__ void scanBlockBelloch(unsigned int* addresses){

    for (int d = 2; d <= BLK_DIM; d = d*2) {   
        __syncthreads();  
        if (THID % d == d-1)  
            addresses[THID] += addresses[THID-d/2];  
    }

    if(THID == (BLK_DIM-1)) {
        addresses[THID] = 0;
    }

    for(int d=BLK_DIM; d > 1; d/=2){
        __syncthreads();
        if(THID % d == d-1){
            unsigned int val = addresses[THID-d/2];
            addresses[THID-d/2] = addresses[THID];
            addresses[THID] += val;
        }
    }
    __syncthreads();
}

__device__ void scanWarpHillis(unsigned int* addresses){
    __syncwarp();  
    int lane_id = THID%32;
    int initVal = addresses[lane_id];

    for (unsigned int d = 1; d < WARP_SIZE; d = d*2) {
        unsigned int newVal = addresses[lane_id];   
        if (int(lane_id - d) >= 0)  
            newVal += addresses[lane_id-d];  
        __syncwarp();  
        addresses[lane_id] = newVal;
        __syncwarp();  
    }
        //Hillis-Steele Scan gives inclusive scan.
        //to get exclusive scan, subtract the initial values.
    addresses[lane_id] -= initVal;
    __syncwarp();
}

__device__ void scanWarpBelloch(unsigned int* addresses){
    unsigned int lane_id = THID%WARP_SIZE;
    for (int d = 2; d <= WARP_SIZE; d = d*2) {   
        __syncwarp();  
        if (lane_id % d == d-1)  
            addresses[lane_id] += addresses[lane_id-d/2];  
    }

    if(lane_id == (WARP_SIZE-1)) {
        addresses[lane_id] = 0;
    }

    for(int d=WARP_SIZE; d > 1; d/=2){
        __syncwarp();
        if(lane_id % d == d-1){
            unsigned int val = addresses[lane_id-d/2];
            addresses[lane_id-d/2] = addresses[lane_id];
            addresses[lane_id] += val;
        }
    }
    __syncwarp();
}

__device__ void compactBlock(unsigned int *degrees, unsigned int V, Node** tail, Node** head, unsigned int* bufTailPtr, unsigned int level){

    unsigned int glThreadIdx = blockIdx.x * BLK_DIM + THID; 
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    __shared__ unsigned int bTail;
    
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + glThreadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;

        addresses[THID] = predicate[THID];

        scanBlock(addresses);

        
        if(THID == BLK_DIM - 1){  
            int nv =  addresses[THID] + predicate[THID];            
            bTail = nv>0? atomicAdd(bufTailPtr, nv) : 0;
            
            if(allocationRequired(tail[0], bTail+nv)) // adding nv since bTail is old value of bufTail
                allocateMemory(tail, head);
        }

        // this sync is necessary so that memory is allocated before writing to buffer
        __syncthreads();
        
        addresses[THID] += bTail;
        
        if(predicate[THID])
            writeToBuffer(tail[0], addresses[THID], v);
        
        __syncthreads();
            
    }
}

__device__ void compactWarp(unsigned int* temp, unsigned int* addresses, unsigned int* predicate, 
                            Node** tail, Node** head, unsigned int* bufTailPtr, 
                            volatile unsigned int* lock){
    
    // __syncwarp();

    unsigned int lane_id = THID%WARP_SIZE;

    unsigned int bTail;
    
    addresses[lane_id] = predicate[lane_id];

    scanWarp(addresses);
    // todo: look for atomic add at warp level.
    
    if(lane_id == WARP_SIZE-1){
        unsigned int nv = addresses[lane_id]+predicate[lane_id]; // nv can be zero if no vertex was found in this warp
        bTail = nv>0? atomicAdd(bufTailPtr, nv) : 0;
        if(allocationRequired(tail[0], bTail+nv)){ // adding nv since bTail is old value of bufTail
            // printf("Req %d", THID);
            atomicCAS((unsigned int*)lock, 2, 0); // resets the lock in case a memory was allocated before
            __threadfence_block();
            allocateMemoryMutex(tail, head, lock);
        }   
    }  
    
    bTail = __shfl_sync(0xFFFFFFFF, bTail, WARP_SIZE-1);
    
    addresses[lane_id] += bTail;
    
    if(predicate[lane_id])
        writeToBuffer(tail[0], addresses[lane_id], temp[lane_id]);
        
        // reset for next iteration
    predicate[lane_id] = 0;

        
    // __syncwarp();
}

// __device__ inline unsigned int getWriteLoc(unsigned int* bufTail){
//     return atomicAdd(bufTail, 1);
// }


__device__ void writeToBuffer(Node* tail, unsigned int loc, unsigned int v){
    if(loc < tail->limit - BUFF_SIZE){ // write to prev node
        tail->prev->data[loc%BUFF_SIZE] = v;
    }else{    // write to current node
        tail->data[loc%BUFF_SIZE] = v;
    }
}

__device__ void advanceNode(Node** head){
    Node* temp = head[0];
    head[0] = head[0]->next;
    free(temp);
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

__device__ void syncBlocks(volatile unsigned int* blockCounter){
    
    if (THID==0)
    {
        atomicAdd((unsigned int*)blockCounter, 1);
        __threadfence();
        while(blockCounter[0]<BLK_NUMS){
            // number of blocks can't be greater than SMs, else it'll cause infinite loop... 
            // printf("%d ", blockCounter[0]);
        };// busy wait until all blocks increment
    }   
    __syncthreads();
}

__device__ unsigned int ldg (const unsigned int * p)
{
    unsigned int out;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, volatile unsigned int* blockCounter){
    
    
    __shared__ Node* tail;
    __shared__ Node* head;
    __shared__ unsigned int bufTail;
    __shared__ unsigned int base;
    __shared__ unsigned int predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    __shared__ volatile unsigned int lock;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;

    tail = NULL;
    head = NULL;
    bufTail = 0;
    base = 0;
    predicate[THID] = 0;
    lock = 0;

    compactBlock(d_p.degrees, V, &tail, &head, &bufTail, level);
    if(level == 1 && THID == 0) printf("%d ", bufTail);

    __syncthreads();

    // bufTail is being incremented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose base is introduced, is incremented whenever a warp takes a job.
    
    // todo: busy waiting on several blocks

    syncBlocks(blockCounter);
    // bufTail = 10;
    // for(unsigned int i = warp_id; i<bufTail ; i += WARPS_EACH_BLK){
    // this for loop is a wrong choice, as many threads might exit from the loop checking the condition     
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads, so can't put after break or continue...
        if(base == bufTail) break;
        i = base + warp_id;
        
        if(THID == 0){
            assert(head!=NULL);
            if(base >= head->limit){
                advanceNode(&head);
            }
            base += WARPS_EACH_BLK;
            if(bufTail < base )
                base = bufTail;
        }
        __syncthreads();
        if(i >= bufTail) continue; // this warp won't have to do anything 

        
        unsigned int v = readFromBuffer(head, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];
        unsigned int b1 = start;
        // for(int j = start + lane_id; j<end ; j+=32){
        // the for loop may leave some of the threads inactive in its last iteration
        // following while loop will keep all threads active until the continue condition
        while(true){
            __syncwarp();

            compactWarp(temp+(warp_id*WARP_SIZE), addresses+(warp_id*WARP_SIZE), predicate+(warp_id*WARP_SIZE), &tail, &head, &bufTail, &lock);
            __syncwarp();

            if(b1 >= end) break;

            unsigned int j = b1 + lane_id;
            b1 += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(ldg(d_p.degrees+u) > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
                
                if(a == level+1){
                    temp[THID] = u;
                    predicate[THID] = 1;
                    // unsigned int loc = getWriteLoc(&bufTail);
                    // writeToBuffer(shBuffer, &glBuffer, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
                // __threadfence();
            }
        }        
    }
    
    __syncthreads();

    if(THID == 0 && bufTail!=0){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
        if(head!=NULL) free(head);
    }

}


