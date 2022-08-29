
#include "../inc/host_funcs.h"

void processNode(unsigned int v, Graph &g, unsigned int* buffer, unsigned int &tail){

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

void find_kcore(string data_file,bool write_to_disk){
    cout<<"start loading graph file from disk to memory..."<<endl;    
    Graph data_graph(data_file);
    cout<<"graph loading complete..."<<endl;
    unsigned int buffer[5e6];

    unsigned int level = 0;
    unsigned int count = 0;

    for(unsigned int level=0; count<data_graph.V; level++){
        unsigned int tail = 0;

        for(int i=0;i<data_graph.V;i++)
            processNode(i, data_graph, buffer, tail)
        

        for(int i=0;i<tail;i++)
            processNode(i, data_graph, buffer, tail)
            
        count+=tail;
    }

}
