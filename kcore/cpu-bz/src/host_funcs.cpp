
#include "../inc/host_funcs.h"
#include <chrono>
void processNode(unsigned int v, Graph &g, unsigned int* buffer, unsigned int &tail, unsigned int level){

    unsigned int start = g.neighbors_offset[v];
    unsigned int end = g.neighbors_offset[v+1];

    for(unsigned int j = start; j<end; j++){
        
        unsigned int u = g.neighbors[j];

        if(g.degrees[u] > level){
            g.degrees[u]--; 
            
            if(g.degrees[u]==level){
                // add to buffer
                buffer[tail++] = u;
            }
            
        }

    }
           
}

void find_kcore(string data_file, bool write_to_disk){
    cout<<"start loading graph file from disk to memory..."<<endl;    
    Graph G(data_file);
    cout<<"graph loading complete..."<<endl;

    // BZ Algorithm implementation 

    auto start = chrono::steady_clock::now();
    unsigned int md = 0;
    for(int v=0;v<G.V;v++){
        md = max(G.degrees[v], md);
    }
    
    unsigned int bin[md+1] = {0}; // initializes all zeros.
    for(int v=0;v<G.V;v++){
        bin[G.degrees[v]]++;
    }

    bin[0] = 1;
    partial_sum(bin, bin+md, bin+1);

    unsigned int pos[G.E+G.V];
    unsigned int vert[G.E+G.V];
    for(int v=0;v<G.V;v++){
        pos[v] = bin[G.degrees[v]];
        vert[pos[v]] = v;
        bin[G.degrees[v]]++;
    }
    cout<<"Max degree "<<md<<endl;

    for(unsigned int d=md;d>0;d--){
        bin[d] = bin[d-1];
    }
    bin[0] = 0;

    for(unsigned int i=0;i<G.V;i++){
        unsigned int v = vert[i];
        unsigned int j = G.neighbors_offset[v];
        unsigned int end = G.neighbors_offset[v+1];
        for (;j<end;j++){
            unsigned int u=G.neighbors[j];
            if(G.degrees[u]>G.degrees[v]){
                unsigned int du = G.degrees[u];
                unsigned int pu = pos[u];
                unsigned int pw = bin[du];
                unsigned int w = vert[pw];
                if(u!=w){
                    pos[u] = pw; vert[pu] = w;
                    pos[w] = pu; vert[pw] = u;
                }
                bin[du]++;
                G.degrees[u]--;
            }
        }
    }
    


    auto end = chrono::steady_clock::now();
    cout << "Elapsed Time: "
    << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;

    
    if(write_to_disk){
        cout<<"Writing kcore to disk started... "<<endl;
        write_kcore_to_disk(G.degrees, G.V, data_file);
        cout<<"Writing kcore to disk completed... "<<endl;
    }

}
