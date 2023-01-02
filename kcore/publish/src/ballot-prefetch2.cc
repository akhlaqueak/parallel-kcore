

__global__ void selectNodesAtLevel62(unsigned int *degrees, unsigned int level, unsigned int V, 
                 unsigned int* bufTails, unsigned int* glBuffers){

    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];
    __shared__ volatile unsigned int addresses[BLK_DIM];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    if(THID==0){
        bufTail = 0;
        glBuffer = glBuffers+(blockIdx.x*GLBUFFER_SIZE);
    }

    unsigned int glThreadIdx = blockIdx.x * BLK_DIM + THID; 

    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + glThreadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;
        if(predicate[THID]) temp[THID] = v;

        compactWarpBallot(predicate, addresses, temp, glBuffer, &bufTail);        
        
        __syncthreads();
            
    }

    if(THID==0){
        bufTails[blockIdx.x] = bufTail;
    }
}







__global__ void processNodes62(G_pointers d_p, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers, 
                    unsigned int *global_count){
    __shared__ unsigned int bufTail;
    __shared__ unsigned int *glBuffer;
    __shared__ unsigned int base;
    // __shared__ unsigned int initTail;
    __shared__ unsigned int prefv[WARPS_EACH_BLK];
    __shared__ unsigned int prefst[WARPS_EACH_BLK];
    __shared__ unsigned int prefen[WARPS_EACH_BLK];
    __shared__ bool bpref[WARPS_EACH_BLK];

    __shared__ unsigned int C1[MAX_PREF * 31];
    __shared__ unsigned int C2[MAX_PREF * 31];
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
            if (lane_id == 0)
            {
                v = prefv[warp_id] = readFromBuffer(glBuffer, warp_id - 1);
                st = prefst[warp_id] = d_p.neighbors_offset[v];
                en = prefen[warp_id] = d_p.neighbors_offset[v + 1];
                pref = bpref[warp_id] = en - st <= MAX_PREF;
            }
            // v=__shfl_sync(FULL, v, 0);
            // st=__shfl_sync(FULL, st, 0);
            // en=__shfl_sync(FULL, en, 0);
            // pref=__shfl_sync(FULL, pref, 0);
            // for (unsigned int i = st+lane_id, j = lane_id; i < en && pref; i += 32, j += 32)
            // {
            //     wrBuff[(warp_id-1) * MAX_PREF + j] = d_p.neighbors[i];
            // }

            __syncwarp();
            for (unsigned int i = prefst[warp_id] + lane_id, j = lane_id; i < prefen[warp_id] && bpref[warp_id]; i += 32, j += 32)
            {
                wrBuff[(warp_id - 1) * MAX_PREF + j] = d_p.neighbors[i];
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
        if (base == bufTail)
            break; // all the threads will evaluate to true at same iteration

        swapBuffers(&wrBuff, &rdBuff); // swaps reading and writing buffer pointers.

        // warp0 will also read following value, but it's none of use
        // reading it only for simplicity of code, else need a condition
        v = prefv[warp_id];
        st = prefst[warp_id];
        en = prefen[warp_id];
        pref = bpref[warp_id];
        // npref, bufTail are shared mem variables... 
        regnpref = npref;
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

            for (int i = 0; i < npref; i++)
            {
                if (lane_id == 0)
                { // running only first lane, because shared memory operation is inside
                    v = prefv[i + 1] = readFromBuffer(glBuffer, base + i);
                    st = prefst[i + 1] = d_p.neighbors_offset[v];
                    en = prefen[i + 1] = d_p.neighbors_offset[v + 1];
                    pref = bpref[i + 1] = en - st <= MAX_PREF;
                }
                // v=__shfl_sync(FULL, v, 0);
                // st=__shfl_sync(FULL, st, 0);
                // en=__shfl_sync(FULL, en, 0);
                // pref=__shfl_sync(FULL, pref, 0);
                // for (unsigned int k = st+lane_id, j = lane_id; k < en && pref; k += 32, j += 32)
                // {
                //     wrBuff[i * MAX_PREF + j] = d_p.neighbors[k];
                // }

                __syncwarp();
                for (unsigned int k = prefst[i + 1] + lane_id, j = lane_id; k < prefen[i + 1] && bpref[i + 1]; k += 32, j += 32)
                {
                    wrBuff[i * MAX_PREF + j] = d_p.neighbors[k];
                }
            }

            continue; // warp0 doesn't process nodes.
        }
        bool pred = false;

        for (unsigned int j = st, k = lane_id; j < en; j += 32, k += 32)
        {
            unsigned int jl = j + lane_id;
            if (jl >= en)
                break;
            unsigned int u = pref ? rdBuff[(warp_id - 1) * MAX_PREF + k] : d_p.neighbors[jl];


            unsigned int loc = scanIndexBallot(pred, &bufTail);
            if(pred){
                writeToBuffer(glBuffer, loc, u);
            }

            pred = false;

            if (d_p.degrees[u] > level)
            {

                unsigned int a = atomicSub(d_p.degrees + u, 1);

                pred = (a==level+1);

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



int kcoreBallotScanPrefetch2(Graph &data_graph){

    G_pointers data_pointers;

    malloc_graph_gpu_memory(data_graph, data_pointers);

    unsigned int level = 0;
    unsigned int count = 0;
    unsigned int* global_count  = NULL;
    unsigned int* bufTails  = NULL;
    unsigned int* glBuffers     = NULL;

    chkerr(cudaMalloc(&global_count, sizeof(unsigned int)));
    chkerr(cudaMalloc(&bufTails, sizeof(unsigned int)*BLK_NUMS));
    cudaMemset(global_count, 0, sizeof(unsigned int));
    chkerr(cudaMalloc(&glBuffers,sizeof(unsigned int)*BLK_NUMS*GLBUFFER_SIZE));
       
    
	// cout<<"K-core Computation Started";

    auto start = chrono::steady_clock::now();
    while(count < data_graph.V){
        cudaMemset(bufTails, 0, sizeof(unsigned int)*BLK_NUMS);

        selectNodesAtLevel62<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, level, 
                        data_graph.V, bufTails, glBuffers);

        processNodes62<<<BLK_NUMS, BLK_DIM>>>(data_pointers, level, data_graph.V, 
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
