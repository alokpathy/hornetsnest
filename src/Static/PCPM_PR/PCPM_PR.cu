#include "Static/PCPM_PR/PCPM_PR.cuh"

#include "math.h"
#include <vector>

namespace hornets_nest {

PCPM_PR::PCPM_PR(HornetGraph& hornet) : 
                        StaticAlgorithm(hornet),
                        vqueue(hornet),
                        src_equeue(hornet),
                        dst_equeue(hornet),
                        load_balancing(hornet) {
                        // nodes_removed(hornet) {}
   
    // Very space inefficient. TODO: fix this.
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        hornet_csr_off[i]   = new vid_t[hornet.nV() + 1]();
        hornet_csr_edges[i] = new vid_t[hornet.nE()]();

        memset(hornet_csr_off[i], 0, (hornet.nV() + 1) * sizeof(vid_t));
    } 

    std::cout << "nv: " << hornet.nV() << "\n";
    gpu::allocate(pr, hornet.nV());
}

struct FindNeighbors {
    TwoLevelQueue<vid_t> src_equeue;
    TwoLevelQueue<vid_t> dst_equeue;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vid_t dst = e.dst_id();

        src_equeue.insert(src);
        dst_equeue.insert(dst);
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
    TwoLevelQueue<MsgData> **msg_queue;
    uint32_t vertices_per_part;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t s = v.id();
        vid_t t = e.dst_id();

        MsgData msg;
        msg.val = pr[s] / v.degree();
        msg.dst = t;
    
        uint32_t partition = t / vertices_per_part;

        printf("%d\n", partition);
        msg_queue[partition]->insert(msg);
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
             TwoLevelQueue<MsgData> **msg_queue,
             uint32_t start_vertex, 
             uint32_t end_vertex,
             TwoLevelQueue<vid_t> vqueue,
             float *pr,
             uint32_t v_per_part,
             load_balancing::VertexBased1 load_balancing) {
    
    
    for (uint32_t i = start_vertex; i < end_vertex; i++) {
        vqueue.insert(i);
    }

    forAllEdges(hornet, vqueue, ScatterPR { pr, msg_queue, v_per_part }, 
                load_balancing);
}

void PCPM_PR::run() {

    // Create new HornetGraph instance per partition.
    HornetGraph **hornets = new HornetGraph*[NUM_PARTS];
    // TwoLevelQueue<MsgData> msg_queue[NUM_PARTS];
    TwoLevelQueue<MsgData> **msg_queue = new TwoLevelQueue<MsgData>*[NUM_PARTS];
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        hornets[i] = hornet_init(hornet, hornet_csr_off[i], 
                                  hornet_csr_edges[i]);

        msg_queue[i] = new TwoLevelQueue<MsgData>(hornet);
    }

    // msg_queue[0]->insert(MsgData { 0.123f, 1283 } );
    std::cout << "msgq size: " << msg_queue[0]->size() << std::endl;

    // Populate each HornetGraph instance.
    uint32_t vertices_per_part = hornet.nV() / NUM_PARTS;
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        uint32_t start_vertex = i * vertices_per_part;
        uint32_t end_vertex = (i + 1) * vertices_per_part;
        if (end_vertex > hornet.nV()) {
            end_vertex = hornet.nV();
        }

        for (uint32_t j = start_vertex; j < end_vertex; j++) {
            vqueue.insert(j);            
        }

        forAllEdges(hornet, vqueue, FindNeighbors { src_equeue, dst_equeue },
                    load_balancing);

        src_equeue.swap();
        dst_equeue.swap();

        gpu::BatchUpdate batch_update_src(
                           (vid_t*) src_equeue.device_input_ptr(),
                           (vid_t*) dst_equeue.device_input_ptr(),
                                    src_equeue.size());


        hornets[i]->insertEdgeBatch(batch_update_src);
        vqueue.swap();
        src_equeue.swap();
        dst_equeue.swap();
    }

    // Initialize pagerank values.
    forAllVertices(hornet, ComputePR { pr });
    
    for (uint32_t i = 0; i < NUM_PARTS; i++) {
        uint32_t start_vertex = i * vertices_per_part;
        uint32_t end_vertex = (i + 1) * vertices_per_part;
        if (end_vertex > hornet.nV()) {
            end_vertex = hornet.nV();
        }

        scatter(*hornets[i], msg_queue, start_vertex, end_vertex, vqueue, pr,
                vertices_per_part, load_balancing);
    }
}

PCPM_PR::~PCPM_PR() {
}

void PCPM_PR::reset() {
}

void PCPM_PR::release() {
}

}
