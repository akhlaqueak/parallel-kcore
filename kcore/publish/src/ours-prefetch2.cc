
__global__ void selectNodesAtLevel32(unsigned int *degrees, unsigned int level, unsigned int V,
                                     unsigned int *bufTails, unsigned int *glBuffers)
{

    __shared__ unsigned int *glBuffer;
    __shared__ unsigned int bufTail;

    if (THID == 0)
    {
        bufTail = 0;
        glBuffer = glBuffers + blockIdx.x * GLBUFFER_SIZE;
    }
    __syncthreads();

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int base = 0; base < V; base += N_THREADS)
    {

        unsigned int v = base + global_threadIdx;

        if (v >= V)
            continue;

        if (degrees[v] == level)
        {
            unsigned int loc = atomicAdd(&bufTail, 1);
            writeToBuffer(glBuffer, loc, v);
        }
    }

    __syncthreads();

    if (THID == 0)
    {
        bufTails[blockIdx.x] = bufTail;
    }
}

__device__ void swapBuffers(unsigned int** X, unsigned int** Y){
    auto temp = *X;
    *X = *Y;
    *Y = temp;
}

__global__ void processNodes32(G_pointers d_p, int level, int V,
                               unsigned int *bufTails, unsigned int *glBuffers,
                               unsigned int *global_count)
{
    __shared__ unsigned int bufTail;
    __shared__ unsigned int *glBuffer;
    __shared__ unsigned int base;
    // __shared__ unsigned int initTail;
    __shared__ unsigned int prefv[WARPS_EACH_BLK];
    __shared__ unsigned int prefst[WARPS_EACH_BLK];
    __shared__ unsigned int prefen[WARPS_EACH_BLK];
    __shared__ bool  bpref[WARPS_EACH_BLK];

    __shared__ unsigned int C1[MAX_PREF*31];
    __shared__ unsigned int C2[MAX_PREF*31];
    __shared__ int npref;
    __shared__ unsigned int *wrBuff;
    __shared__ unsigned int *rdBuff;

    unsigned int warp_id = WARPID;
    unsigned int lane_id = LANEID;
    unsigned int regTail, regnpref;
    unsigned int v, st, en;
    bool pref;
    
    if (THID == 0)
    {
        bufTail = bufTails[blockIdx.x];
        // initTail = bufTail;
        base = 0;
        npref = 0;
        glBuffer = glBuffers + blockIdx.x * GLBUFFER_SIZE;
        wrBuff = C1;
        rdBuff = C2;
        assert(glBuffer != NULL);
    }

    __syncthreads();

    // if(THID == 0 && level == 1)
    //     printf("%d ", bufTail);
    // 0-th iteration
    // in the first iteration all warps prefetch their data, as they shouldn't wait warp0 to do a long job
    // from subsequent iteration, warp0 prefetches, and rest of warps process
    if (warp_id > 0)
        if (warp_id - 1 < bufTail)
        {
            if(lane_id==0){
                v = prefv[warp_id] = readFromBuffer(glBuffer, warp_id - 1);
                st = prefst[warp_id] = d_p.neighbors_offset[v];
                en = prefen[warp_id] = d_p.neighbors_offset[v + 1];
                bpref[warp_id] = en - st <= MAX_PREF;
            }
            __syncwarp();

            for (unsigned int i = prefst[warp_id], j = lane_id; i < prefen[warp_id] && bpref[warp_id]; i += 32, j += 32)
            {
                unsigned int il = i + lane_id;
                if (il < prefen[warp_id])
                {
                    wrBuff[(warp_id-1) * MAX_PREF + j] = d_p.neighbors[il];
                }
            }
        }
    if (THID == 0)
    {
        npref = min(WARPS_EACH_BLK - 1, bufTail);
    }

    // if(THID == 0){
    //     base += WARPS_EACH_BLK-1;
    //     if(bufTail < base )
    //         base = bufTail;
    // }

    // bufTail is being incrmented within the loop,
    // warps should process all the nodes added during the execution of loop

    // for(unsigned int i = warp_id; i<bufTail ; i +=warps_each_block ){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while (true)
    {
        __syncthreads(); // syncthreads must be executed by all the threads
        swapBuffers(&wrBuff, &rdBuff); // swaps reading and writing buffer pointers.
        if(warp_id <= npref){
            v = prefv[warp_id];
            st = prefst[warp_id];
            en = prefen[warp_id];
            pref = bpref[warp_id];
        }
        regnpref = npref;
        if (base == bufTail)
            break; // all the threads will evaluate to true at same iteration
        regTail = bufTail;
        __syncthreads();
        if (warp_id > regnpref)
            continue;

        if (warp_id == 0)
        {
            if (lane_id == 0)
            {
                // update base for next iteration
                base += npref;
                npref = min(WARPS_EACH_BLK - 1, regTail - base);                
            }
            __syncwarp(); // so that other lanes can see updated base value

            for(int i=0;i<npref;i++){
                if(lane_id==0){
                    v = prefv[i+1] = readFromBuffer(glBuffer, base+i);
                    st = prefst[i+1] = d_p.neighbors_offset[v];
                    en = prefen[i+1] = d_p.neighbors_offset[v + 1];
                    bpref[i+1] = en - st <= MAX_PREF;
                }
                __syncwarp();
                
                for (unsigned int k = prefst[i+1], j = lane_id; i < prefen[i+1] && bpref[i+1]; k += 32, j += 32)
                {
                    unsigned int kl = k + lane_id;
                    if (kl < prefen[i+1])
                    {
                        wrBuff[i * MAX_PREF + j] = d_p.neighbors[kl];
                    }
                }
            }

            continue; // warp0 doesn't process nodes.
        }


        // while (true)
        // {
        //     __syncwarp();

        //     if (start >= end)
        //         break;

        //     unsigned int j = start + lane_id;
        //     start += WARP_SIZE;
        //     if (j >= end)
        //         continue;
        for(unsigned int j=st, k=lane_id; j<en; j+=32, k+=32){
            unsigned int jl = j+lane_id;
            if(jl>=en) continue;
            unsigned int u = pref? rdBuff[(warp_id-1) * MAX_PREF + k] : d_p.neighbors[jl];            

            if ( d_p.degrees[u] > level)
            {

                unsigned int a = atomicSub(d_p.degrees + u, 1);

                if (a == level + 1)
                {
                    unsigned int loc = atomicAdd(&bufTail, 1);

                    writeToBuffer(glBuffer, loc, u);
                }

                if (a <= level)
                {
                    // node degree became less than the level after decrementing...
                    atomicAdd(d_p.degrees + u, 1);
                }
            }
        }
    }

    if (THID == 0 && bufTail > 0)
    {
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
    }
}

int kcorePrefetch2(Graph &data_graph)
{

    G_pointers data_pointers;

    malloc_graph_gpu_memory(data_graph, data_pointers);

    unsigned int level = 0;
    unsigned int count = 0;
    unsigned int *global_count = NULL;
    unsigned int *bufTails = NULL;
    unsigned int *glBuffers = NULL;

    chkerr(cudaMalloc(&global_count, sizeof(unsigned int)));
    chkerr(cudaMalloc(&bufTails, sizeof(unsigned int) * BLK_NUMS));
    cudaMemset(global_count, 0, sizeof(unsigned int));
    chkerr(cudaMalloc(&glBuffers, sizeof(unsigned int) * BLK_NUMS * GLBUFFER_SIZE));

    // cout<<"K-core Computation Started";

    auto start = chrono::steady_clock::now();
    while (count < data_graph.V)
    {
        cudaMemset(bufTails, 0, sizeof(unsigned int) * BLK_NUMS);

        selectNodesAtLevel32<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, level,
                                                    data_graph.V, bufTails, glBuffers);

        processNodes32<<<BLK_NUMS, BLK_DIM>>>(data_pointers, level, data_graph.V,
                                              bufTails, glBuffers, global_count);

        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }
    // cout<<"Done "<<"Kmax: "<<level-1<<endl;

    auto end = chrono::steady_clock::now();

    cudaFree(glBuffers);
    free_graph_gpu_memory(data_pointers);

    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
}
