
#include "../inc/host_funcs.h"
#include <ctime>
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
    Graph data_graph(data_file);
    cout<<"graph loading complete..."<<endl;
    unsigned int buffer[50000000];

    unsigned int count = 0;
    double tick = time(NULL);
    for(unsigned int level=0; count<data_graph.V; level++){
        unsigned int tail = 0;

        for(int i=0;i<data_graph.V;i++)
            if(data_graph.degrees[i] == level)
                buffer[tail++] = i;
     cout<<"Total nodes here: "<<tail<<endl;
        for(int i=0;i<tail;i++)
            processNode(buffer[i], data_graph, buffer, tail, level);
            
        count+=tail;

        cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;

    }

    double tock = time(NULL);
    cout<<"Elapsed Time: "<<tock-tick<<endl;

    
    if(write_to_disk){
        cout<<"Writing kcore to disk started... "<<endl;
        write_kcore_to_disk(data_graph.degrees, data_graph.V, data_file);
        cout<<"Writing kcore to disk completed... "<<endl;
    }

}
