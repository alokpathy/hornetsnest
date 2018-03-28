#include "Static/PCPM_PR/PCPM_PR.cuh"

#include "math.h"
#include <vector>

namespace hornets_nest {

PCPM_PR::PCPM_PR(HornetGraph& hornet) : 
                        StaticAlgorithm(hornet),
                        vqueue(hornet),
                        // src_equeue(hornet),
                        // dst_equeue(hornet),
                        load_balancing(hornet) {
                        // nodes_removed(hornet) {}
   
    // Very space inefficient. TODO: fix this.
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        hornet_csr_off[i]   = new vid_t[hornet.nV() + 1]();
        hornet_csr_edges[i] = new vid_t[hornet.nE()]();

        memset(hornet_csr_off[i], 0, (hornet.nV() + 1) * sizeof(vid_t));
    } 

    std::cout << "nv: " << hornet.nV() << " ne: " << hornet.nE() << "\n";
    #if 0
    gpu::allocate(pr,                hornet.nV());
    gpu::allocate(hd_data().src,     hornet.nE());
    gpu::allocate(hd_data().dst,     hornet.nE());
    gpu::allocate(hd_data().counter, 1);
    gpu::allocate(hd_data().val,     NUM_PARTS);
    gpu::allocate(hd_data().dstid,   NUM_PARTS);
    gpu::allocate(hd_data().msg_counter,   NUM_PARTS);
    #endif

    cudaMalloc(&pr,                    hornet.nV() * sizeof(float));
    cudaMalloc(&hd_data().src,         hornet.nE() * sizeof(vid_t));
    cudaMalloc(&hd_data().dst,         hornet.nE() * sizeof(vid_t));
    cudaMalloc(&hd_data().counter,     sizeof(int));
    cudaMalloc(&hd_data().val,         NUM_PARTS * sizeof(float*));
    cudaMalloc(&hd_data().dstid,       NUM_PARTS * sizeof(vid_t*));
    cudaMalloc(&hd_data().msg_counter, NUM_PARTS * sizeof(int));

    cudaMemset(hd_data().counter, 0, sizeof(int));
    cudaMemset(hd_data().msg_counter, 0, NUM_PARTS * sizeof(int));
}

struct FindNeighbors {
    // TwoLevelQueue<vid_t> src_equeue;
    // TwoLevelQueue<vid_t> dst_equeue;
    HostDeviceVar<PR_DATA> hd;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vid_t dst = e.dst_id();

        int spot = atomicAdd(hd().counter, 1);
        hd().src[spot] = src;
        hd().dst[spot] = dst;
    }
};

struct ComputePR {
    float *pr;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        pr[id] = (1.0f) / v.degree();
    }
};

struct ScatterPR {
    float *pr;
    // TwoLevelQueue<MsgData> **msg_queue;
    HostDeviceVar<PR_DATA> hd;
    uint32_t vertices_per_part;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t s = v.id();
        vid_t t = e.dst_id();
    
        uint32_t partition = t / vertices_per_part;

        int spot = atomicAdd(&hd().msg_counter[partition], 1);
        hd().val[partition][spot] = pr[s];
        hd().dstid[partition][spot] = t;
    }
};

HornetGraph *hornet_init(HornetGraph &hornet,
                         vid_t *hornet_csr_off,
                         vid_t *hornet_csr_edges) {

    HornetInit hornet_init(hornet.nV(), 0, hornet_csr_off,
                           hornet_csr_edges, false);

    HornetGraph *h_new = new HornetGraph(hornet_init);

    return h_new;
}

void scatter(HornetGraph &hornet, 
             // TwoLevelQueue<MsgData> **msg_queue,
             HostDeviceVar<PR_DATA> hd,
             uint32_t start_vertex, 
             uint32_t end_vertex,
             TwoLevelQueue<vid_t> vqueue,
             float *pr,
             uint32_t v_per_part,
             load_balancing::VertexBased1 load_balancing) {
    
    
    for (uint32_t i = start_vertex; i < end_vertex; i++) {
        vqueue.insert(i);
    }

    std::cout << "before" << std::endl;
    forAllEdges(hornet, vqueue, ScatterPR { pr, hd, v_per_part }, 
                load_balancing);
    std::cout << "after" << std::endl;
}

void PCPM_PR::run() {

    // Create new HornetGraph instance per partition.
    HornetGraph **hornets = new HornetGraph*[NUM_PARTS];
    // TwoLevelQueue<MsgData> msg_queue[NUM_PARTS];
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        hornets[i] = hornet_init(hornet, hornet_csr_off[i], 
                                  hornet_csr_edges[i]);
    }

    // Populate each HornetGraph instance.
    uint32_t vertices_per_part = hornet.nV() / NUM_PARTS;
    float **tmp_val = new float*[NUM_PARTS];
    vid_t **tmp_dst = new vid_t*[NUM_PARTS];
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        uint32_t start_vertex = i * vertices_per_part;
        uint32_t end_vertex = (i + 1) * vertices_per_part;

        std::cout << "start " << start_vertex << " end " << end_vertex << "\n";
        if (end_vertex > hornet.nV()) {
            end_vertex = hornet.nV();
        }

        for (uint32_t j = start_vertex; j < end_vertex; j++) {
            vqueue.insert(j);            
        }

        forAllEdges(hornet, vqueue, FindNeighbors { hd_data }, load_balancing);

        int size = 0;
        cudaMemcpy(&size, hd_data().counter, sizeof(int), cudaMemcpyDeviceToHost);

        gpu::BatchUpdate batch_update(hd_data().src, hd_data().dst, size);

        // Note: this is only correct if the graph is undirected.
        cudaMalloc(&tmp_val[i], size);
        cudaMalloc(&tmp_dst[i], size);

        hornets[i]->insertEdgeBatch(batch_update);
        vqueue.swap();
    }
    cudaMemcpy(hd_data().val, tmp_val, NUM_PARTS * sizeof(float*),
               cudaMemcpyHostToDevice);

    cudaMemcpy(hd_data().dstid, tmp_dst, NUM_PARTS * sizeof(vid_t*),
               cudaMemcpyHostToDevice);

    delete[] tmp_val;
    delete[] tmp_dst;

    // Initialize pagerank values.
    forAllVertices(hornet, ComputePR { pr });
    
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        uint32_t start_vertex = i * vertices_per_part;
        uint32_t end_vertex = (i + 1) * vertices_per_part;
        if (end_vertex > hornet.nV()) {
            end_vertex = hornet.nV();
        }

        scatter(*hornets[i], hd_data, start_vertex, end_vertex, vqueue, pr,
                vertices_per_part, load_balancing);
    }
}

PCPM_PR::~PCPM_PR() {
    cudaFree(pr);
    cudaFree(hd_data().src);
    cudaFree(hd_data().dst);
    cudaFree(hd_data().counter);

    int **tmp = new int*[NUM_PARTS];

    cudaMemcpy(tmp, hd_data().val, NUM_PARTS * sizeof(int), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        cudaFree(tmp[i]);
    }
    cudaFree(hd_data().val);

    cudaMemcpy(tmp, hd_data().dstid, NUM_PARTS * sizeof(int), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        cudaFree(tmp[i]);
    }
    cudaFree(hd_data().dstid);

    cudaMemcpy(tmp, hd_data().msg_counter, NUM_PARTS * sizeof(int), 
               cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        cudaFree(tmp[i]);
    }
    cudaFree(hd_data().msg_counter);
}

void PCPM_PR::reset() {
}

void PCPM_PR::release() {
}

}
