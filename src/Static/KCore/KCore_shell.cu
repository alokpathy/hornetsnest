#include <Device/Util/Timer.cuh>
#include "Static/KCore/KCore.cuh"
#include <fstream>

#include <nvToolsExt.h>

#define INSERT 0
#define DELETE 1

// #define NVTX_DEBUG

#define NEW_KCORE
// #define SHELL

// #include <Device/Primitives/CubWrapper.cuh>

using namespace timer;
namespace hornets_nest {

KCore::KCore(HornetGraph &hornet) : 
                        StaticAlgorithm(hornet),
                        vqueue(hornet),
                        // src_equeue(hornet, 4.0f),
                        // dst_equeue(hornet, 4.0f),
                        peel_vqueue(hornet),
                        active_queue(hornet),
                        iter_queue(hornet),
                        load_balancing(hornet)
                        {

    // h_copy_csr_off   = new vid_t[hornet.nV() + 1]();
    // h_copy_csr_edges = new vid_t[0]();
    
    // memset(h_copy_csr_off, 0, (hornet.nV() + 1) * sizeof(vid_t));

    gpu::allocate(vertex_pres, hornet.nV());
    // gpu::allocate(vertex_color, hornet.nV());
    // gpu::allocate(vertex_subg, hornet.nV());
    gpu::allocate(vertex_deg, hornet.nV());
    gpu::allocate(vertex_shell, hornet.nV());
    gpu::allocate(hd_data().src,    hornet.nE());
    gpu::allocate(hd_data().dst,    hornet.nE());
    // gpu::allocate(hd_data().src_tot,    hornet.nE());
    // gpu::allocate(hd_data().dst_tot,    hornet.nE());
    gpu::allocate(hd_data().counter, 1);
    // gpu::allocate(hd_data().counter_tot, 1);
    // gpu::memsetZero(hd_data().counter_tot);  // initialize counter for all edge mapping.
}

KCore::~KCore() {
    gpu::free(vertex_pres);
    // gpu::free(vertex_color);
    // gpu::free(vertex_subg);
    gpu::free(vertex_deg);
    gpu::free(vertex_shell);
    gpu::free(hd_data().src);
    gpu::free(hd_data().dst);
    // gpu::free(hd_data().src_tot);
    // gpu::free(hd_data().dst_tot);
    gpu::free(hd_data().counter);
    // gpu::free(hd_data().counter_tot);
    // delete[] h_copy_csr_off;
    // delete[] h_copy_csr_edges;
}

struct PrintVertices {
    const vid_t *src_ptr;
    const vid_t *dst_ptr;
    int32_t size;

    OPERATOR(Vertex &v) {
        if (v.id() == 0) {
            for (uint32_t i = 0; i < size; i++) {
                // printf("%d ", src_ptr[i]);
                printf("batch_src[%u] = %d; batch_dst[%u] = %d;\n", i, src_ptr[i], i,
                                                                   dst_ptr[i]);
            }
        }
    }
};

struct ActiveVertices {
    vid_t *vertex_pres;
    vid_t *deg;
    TwoLevelQueue<vid_t> active_queue;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        if (v.degree() > 0) {
            vertex_pres[id] = 1;
            active_queue.insert(id);
            deg[id] = v.degree();
        }
    }
};

struct PeelVerticesShell {
    vid_t *vertex_pres;
    vid_t *deg;
    vid_t *vertex_shell;
    uint32_t peel;
    TwoLevelQueue<vid_t> peel_queue;
    TwoLevelQueue<vid_t> iter_queue;
    
    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        if (vertex_pres[id] == 1 && deg[id] <= peel) {
            vertex_pres[id] = 2;
            peel_queue.insert(id);
            iter_queue.insert(id);
            vertex_shell[id] = peel;
        }
    }
};

struct RemovePres {
    vid_t *vertex_pres;
    
    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        if (vertex_pres[id] == 2) {
            vertex_pres[id] = 0;
        }
    }
};

struct DecrementDegree {
    vid_t *deg;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vid_t dst = e.dst_id();
        atomicAdd(&deg[src], -1);
        atomicAdd(&deg[dst], -1);
    }
};

struct ExtractSubgraph {
    HostDeviceVar<KCoreData> hd;
    vid_t *vertex_pres;
    
    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vid_t dst = e.dst_id();
        if (vertex_pres[src] == 2 && vertex_pres[dst] == 2) {
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;
        }
    }
};

void KCore::reset() {
    std::cout << "ran1" << std::endl;
}

void oper_bidirect_batch(HornetGraph &hornet, vid_t *src, vid_t *dst, 
                         int size, uint8_t op) {
#ifdef NVTX_DEBUG
    nvtxRangeId_t id3 = nvtxRangeStartA("batch src construct range");
#endif
    gpu::BatchUpdate batch_update(src, dst, size, gpu::BatchType::DEVICE);
    // batch_update.print();

#ifdef NVTX_DEBUG
    nvtxRangeId_t id4 = nvtxRangeStartA("batch insert/delete range");
#endif
    if (op == DELETE) {
        // Delete edges in the forward direction.
        // hornet.deleteEdgeBatch(batch_update_src);
        // hornet.deleteEdgeBatch(batch_update, gpu::batch_property::IN_PLACE);
        hornet.deleteEdgeBatch(batch_update, gpu::batch_property::IN_PLACE);
        // hornet.deleteEdgeBatch(batch_update);
    } else if (op == INSERT) {
        // Delete edges in the forward direction.
        // hornet.insertEdgeBatch(batch_update_src);
        // hornet.insertEdgeBatch(batch_update, gpu::batch_property::IN_PLACE);
        hornet.insertEdgeBatch(batch_update, gpu::batch_property::IN_PLACE);
        // hornet.insertEdgeBatch(batch_update);
    }
#ifdef NVTX_DEBUG
    nvtxRangeEnd(id4);
#endif
}

void kcores_shell(HornetGraph &hornet, 
            HostDeviceVar<KCoreData>& hd, 
            TwoLevelQueue<vid_t> &peel_queue,
            TwoLevelQueue<vid_t> &active_queue,
            TwoLevelQueue<vid_t> &iter_queue,
            load_balancing::VertexBased1 load_balancing,
            vid_t *deg,
            vid_t *vertex_pres,
            vid_t *vertex_shell,
            uint32_t *max_peel) {


    forAllVertices(hornet, ActiveVertices { vertex_pres, deg, active_queue });
    active_queue.swap();

    int n_active = active_queue.size();
    uint32_t peel = 0;

    while (n_active > 0) {
        // std::cout << "peel " << peel << std::endl;
        forAllVertices(hornet, active_queue, 
                PeelVerticesShell { vertex_pres, deg, vertex_shell, peel,
                                    peel_queue, iter_queue} );
        iter_queue.swap();
        // iter_queue.print();
        
        // n_active -= peel_queue.size();
        n_active -= iter_queue.size();
    
        // if (peel_queue.size() == 0) {}
        if (iter_queue.size() == 0) {
            peel++;
            peel_queue.swap();
            if (n_active > 0) {
                forAllVertices(hornet, active_queue, RemovePres { vertex_pres });
            }
        } else {
            forAllEdges(hornet, iter_queue, DecrementDegree { deg }, load_balancing);
        }
    }

    gpu::memsetZero(hd().counter);  // reset counter. 
    peel_queue.swap();
    // peel_queue.print();
    forAllEdges(hornet, peel_queue, 
                    ExtractSubgraph { hd, vertex_pres }, load_balancing);

    *max_peel = peel;
}

void json_dump(vid_t *src, vid_t *dst, uint32_t *peel, uint32_t peel_edges) {
    std::ofstream output_file;
    output_file.open("output.txt");
    
    output_file << "{\n";
    for (uint32_t i = 0; i < peel_edges; i++) {
        output_file << "\"" << src[i] << "," << dst[i] << "\": " << peel[i];
        if (i < peel_edges - 1) {
            output_file << ",";
        }
        output_file << "\n";
    }
    output_file << "}";
    output_file.close();
}

void KCore::run() {
    omp_set_num_threads(72);
    vid_t *shell  = new vid_t[hornet.nV()];

    auto vshell = vertex_shell;
    forAllnumV(hornet, [=] __device__ (int i){ vshell[i] = 0; } );

    uint32_t max_peel = 0;
    kcores_shell(hornet, hd_data, peel_vqueue, active_queue, iter_queue, 
               load_balancing, vertex_deg, vertex_pres, vertex_shell, &max_peel);
    std::cout << "max_peel: " << max_peel << "\n";
    
    cudaMemcpy(shell, vertex_shell, hornet.nV() * sizeof(vid_t), 
                cudaMemcpyDeviceToHost);
    
    for (uint32_t i = 0; i < hornet.nV(); i++) {
        std::cout << shell[i] << "\n";
    }
}

void KCore::release() {
    std::cout << "ran3" << std::endl;
}
}
