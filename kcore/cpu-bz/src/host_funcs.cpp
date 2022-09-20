
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

    auto clkstart = chrono::steady_clock::now();
    unsigned int md = 0;
    for(int v=0;v<G.V;v++){
        md = max(G.degrees[v], md);
    }
    
    unsigned int bin[md+1] = {0}; // initializes all zeros.
    for(int i=0;i<=md;i++)
        bin[i] = 0;
    for(int v=0;v<G.V;v++){
        bin[G.degrees[v]]++;
    }

// 16 for d := 0 to md do begin
// 17 num := bin[d];
// 18 bin[d] := start;
// 19 inc(start, num);
// 20 end;
    unsigned int  start = 1;
    for(unsigned int d=0; d<md;d++){
        unsigned int num = bin[d];
        bin[d] = start;
        start += num;
    }

    unsigned int pos[G.E+G.V];
    unsigned int vert[G.E+G.V];
    // cout<<"Max degree "<<md<<" V: "<<G.V<<" E: "<<G.E<<endl;
    for(int v=0;v<G.V;v++){
        pos[v] = bin[G.degrees[v]];
        // cout<<pos[v]<<endl;
        vert[pos[v]] = v;
        bin[G.degrees[v]]++;
    }

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
    << chrono::duration_cast<chrono::milliseconds>(end - clkstart).count() << endl;

    
    if(write_to_disk){
        cout<<"Writing kcore to disk started... "<<endl;
        write_kcore_to_disk(G.degrees, G.V, data_file);
        cout<<"Writing kcore to disk completed... "<<endl;
    }

}
