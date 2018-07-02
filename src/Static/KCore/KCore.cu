#include <Device/Util/Timer.cuh>
#include "Static/KCore/KCore.cuh"
#include <fstream>

#include <nvToolsExt.h>

#define INSERT 0
#define DELETE 1

// #define NVTX_DEBUG

#define NEW_KCORE
#define SHELL

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

    h_copy_csr_off   = new vid_t[hornet.nV() + 1]();
    h_copy_csr_edges = new vid_t[0]();
    
    memset(h_copy_csr_off, 0, (hornet.nV() + 1) * sizeof(vid_t));

    gpu::allocate(vertex_pres, hornet.nV());
    gpu::allocate(vertex_color, hornet.nV());
    gpu::allocate(vertex_subg, hornet.nV());
    gpu::allocate(vertex_deg, hornet.nV());
    gpu::allocate(hd_data().src,    hornet.nE());
    gpu::allocate(hd_data().dst,    hornet.nE());
    gpu::allocate(hd_data().src_tot,    hornet.nE());
    gpu::allocate(hd_data().dst_tot,    hornet.nE());
    gpu::allocate(hd_data().counter, 1);
    gpu::allocate(hd_data().counter_tot, 1);
    gpu::memsetZero(hd_data().counter_tot);  // initialize counter for all edge mapping.
}

KCore::~KCore() {
    gpu::free(vertex_pres);
    gpu::free(vertex_color);
    gpu::free(vertex_subg);
    gpu::free(hd_data().src);
    gpu::free(hd_data().dst);
    gpu::free(hd_data().src_tot);
    gpu::free(hd_data().dst_tot);
    gpu::free(hd_data().counter);
    gpu::free(hd_data().counter_tot);
    delete[] h_copy_csr_off;
    delete[] h_copy_csr_edges;
}

void KCore::set_hcopy(HornetGraph *h_copy_arg) {
    h_copy_ptr = h_copy_arg;
}

struct CheckDeg {
    TwoLevelQueue<vid_t> vqueue;
    TwoLevelQueue<vid_t> peel_vqueue;
    vid_t *vertex_pres;
    vid_t *vertex_color;
    uint32_t peel;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();

        if (vertex_pres[id] && v.degree() <= peel) {
            vqueue.insert(id);
            peel_vqueue.insert(id);
            vertex_pres[id] = 0;
            vertex_color[id] = 1;
        }
    } 
};

struct SetPresent {
    vid_t *vertex_pres;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        vertex_pres[id] = 1;
    }
};

struct SetColor {
    vid_t *vertex_color;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        vertex_color[id] = 0;
    }
};

struct ClearHCopy {
    HostDeviceVar<KCoreData> hd;
    
    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        auto dst = e.dst_id();

        int spot = atomicAdd(hd().counter, 1);
        hd().src[spot] = src;
        hd().dst[spot] = dst;
    }
};

struct PeelVertices {
    HostDeviceVar<KCoreData> hd;
    vid_t *vertex_color;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        auto dst = e.dst_id();

        #if 0
        if (vertex_color[src] && vertex_color[dst]) {
            if (src < dst) {
                int spot = atomicAdd(hd().counter, 1);
                hd().src[spot] = src;
                hd().dst[spot] = dst;

                int spot_rev = atomicAdd(hd().counter, 1);
                hd().src[spot_rev] = dst;
                hd().dst[spot_rev] = src;
            }
        } else if (vertex_color[src] || vertex_color[dst]) {
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;

            int spot_rev = atomicAdd(hd().counter, 1);
            hd().src[spot_rev] = dst;
            hd().dst[spot_rev] = src;
        }
        #endif
        
        int cond1 = vertex_color[src] && vertex_color[dst];
        int cond2 = (cond1==0) && (vertex_color[src] || vertex_color[dst]); 

        if ((cond1 && src < dst) || cond2) {
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;

            int spot_rev = atomicAdd(hd().counter, 1);
            hd().src[spot_rev] = dst;
            hd().dst[spot_rev] = src;

	}
        #if 0
        int spot = atomicAdd(hd().counter, 1);
        hd().src[spot] = src;
        hd().dst[spot] = dst;

        int spot_rev = atomicAdd(hd().counter, 1);
        hd().src[spot_rev] = dst;
        hd().dst[spot_rev] = src;
        #endif
    }
};

struct Subgraph {
    HostDeviceVar<KCoreData> hd;
    uint32_t peel_edges;
    vid_t *vertex_subg;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        auto dst = e.dst_id();

        if (src < dst && vertex_subg[dst] == 1) {
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;

            int spot_rev = atomicAdd(hd().counter, 1);
            hd().src[spot_rev] = dst;
            hd().dst[spot_rev] = src;

            int spot_tot = atomicAdd(hd().counter_tot, 1);
            // uint32_t spot_tot = peel_edges + spot;
            hd().src_tot[spot_tot] = src;
            hd().dst_tot[spot_tot] = dst;
        }
    }
};

struct SubgraphVertices {
    vid_t *vertex_subg;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vertex_subg[src] = 1;
    }
};

struct ClearSubgraph {
    vid_t *vertex_subg;

    OPERATOR(Vertex &v) {
        vid_t src = v.id();
        vertex_subg[src] = 0;
    }
};

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

struct PeelVerticesNew {
    vid_t *vertex_pres;
    vid_t *deg;
    uint32_t peel;
    TwoLevelQueue<vid_t> peel_queue;
    TwoLevelQueue<vid_t> iter_queue;
    
    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        if (vertex_pres[id] == 1 && deg[id] <= peel) {
            vertex_pres[id] = 2;
            peel_queue.insert(id);
            iter_queue.insert(id);
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

struct GetDegOne {
    TwoLevelQueue<vid_t> vqueue;
    vid_t *vertex_color;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        if (v.degree() == 1) {
            vqueue.insert(id);
            vertex_color[id] = 1;
        }
    }
};

struct DegOneEdges {
    HostDeviceVar<KCoreData> hd;
    vid_t *vertex_color;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vid_t dst = e.dst_id();

        // if (v.degree() == 1) {
        if (vertex_color[src] || vertex_color[dst]) {
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;

            if (!vertex_color[src] || !vertex_color[dst]) {
                int spot_rev = atomicAdd(hd().counter, 1);
                hd().src[spot_rev] = dst;
                hd().dst[spot_rev] = src;
            }
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

void kcores(HornetGraph &hornet, 
            HornetGraph &h_copy,
            TwoLevelQueue<vid_t> &vqueue, 
            HostDeviceVar<KCoreData>& hd, 
            TwoLevelQueue<vid_t> &peel_vqueue,
            load_balancing::VertexBased1 load_balancing,
            uint32_t *max_peel,
            vid_t *vertex_pres,
            vid_t *vertex_subg,
            vid_t *vertex_color,
            uint32_t *ne,
            uint32_t peel_edges) {

    uint32_t peel = 0;
    uint32_t nv = hornet.nV();
    int size = 0;
    
#ifdef NVTX_DEBUG
    nvtxRangeId_t id1 = nvtxRangeStartA("iteration range");
#endif
    while (nv > 0) {
#ifdef NVTX_DEBUG
        nvtxRangeId_t id_init = nvtxRangeStartA("init range");
#endif
        forAllVertices(hornet, SetColor { vertex_color });
        forAllVertices(hornet, CheckDeg { vqueue, peel_vqueue, 
                                          vertex_pres, vertex_color, peel });
#ifdef NVTX_DEBUG
        nvtxRangeEnd(id_init);
#endif
        
        vqueue.swap();
        nv -= vqueue.size();
        
        // vqueue.print();

        if (vqueue.size() > 0) {
            // Find all vertices with degree <= peel.
            gpu::memsetZero(hd().counter);  // reset counter. 

#ifdef NVTX_DEBUG
            nvtxRangeId_t id_peel = nvtxRangeStartA("peel range");
#endif
            forAllEdges(hornet, vqueue, PeelVertices { hd, vertex_color }, 
                        load_balancing); 
#ifdef NVTX_DEBUG
            nvtxRangeEnd(id_peel);
#endif

            cudaMemcpy(&size, hd().counter, sizeof(int), cudaMemcpyDeviceToHost);

            if (size > 0) {
#ifdef NVTX_DEBUG
                nvtxRangeId_t id2 = nvtxRangeStartA("batch range");
#endif
                oper_bidirect_batch(hornet, hd().src, hd().dst, size, DELETE);
                oper_bidirect_batch(h_copy, hd().src, hd().dst, size, INSERT);
#ifdef NVTX_DEBUG
                nvtxRangeEnd(id2);
#endif
            }

            // *ne -= 2 * size;
            *ne -= size;

            vqueue.clear();
        } else {
            peel++;    
            peel_vqueue.swap();
        }
    }
    *max_peel = peel;

    peel_vqueue.swap();

    forAllEdges(h_copy, peel_vqueue, SubgraphVertices { vertex_subg }, load_balancing);

    gpu::memsetZero(hd().counter);  // reset counter. 
    // forAllEdges(h_copy, peel_vqueue, Subgraph { hd, vertex_subg }, load_balancing);
    forAllEdges(h_copy, peel_vqueue, Subgraph { hd, peel_edges, vertex_subg }, 
                load_balancing);

    forAllVertices(h_copy, ClearSubgraph { vertex_subg });
    
    cudaMemcpy(&size, hd().counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (size > 0) {
        oper_bidirect_batch(h_copy, hd().src, hd().dst, size, DELETE);
        // oper_bidirect_batch(h_copy, batch_update, DELETE);
    }
#ifdef NVTX_DEBUG
    nvtxRangeEnd(id1);
#endif
}

void kcores_new(HornetGraph &hornet, 
            HostDeviceVar<KCoreData>& hd, 
            TwoLevelQueue<vid_t> &peel_queue,
            TwoLevelQueue<vid_t> &active_queue,
            TwoLevelQueue<vid_t> &iter_queue,
            load_balancing::VertexBased1 load_balancing,
            vid_t *deg,
            vid_t *vertex_pres,
            uint32_t *max_peel,
            int *batch_size) {

    forAllVertices(hornet, ActiveVertices { vertex_pres, deg, active_queue });
    active_queue.swap();

    int n_active = active_queue.size();
    uint32_t peel = 0;

    while (n_active > 0) {
        // std::cout << "peel " << peel << std::endl;
        forAllVertices(hornet, active_queue, 
                PeelVerticesNew { vertex_pres, deg, peel, peel_queue, iter_queue} );
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
    int size = 0;
    cudaMemcpy(&size, hd().counter, sizeof(int), cudaMemcpyDeviceToHost);
    *batch_size = size;
}

HornetGraph* hornet_copy(HornetGraph &hornet,
                         vid_t *h_copy_csr_off,
                         vid_t *h_copy_csr_edges) {
                         // TwoLevelQueue<vid_t> tot_src_equeue,
                         // TwoLevelQueue<vid_t> tot_dst_equeue) {

    HornetInit hornet_init(hornet.nV(), 0, h_copy_csr_off,
                           h_copy_csr_edges, false);

    HornetGraph *h_copy = new HornetGraph(hornet_init);

    return h_copy;
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

#if !defined SHELL && !defined NEW_KCORE
void KCore::run() {

    omp_set_num_threads(72);
    vid_t *src     = new vid_t[hornet.nE() / 2 + 1];
    vid_t *dst     = new vid_t[hornet.nE() / 2 + 1];
    uint32_t len = hornet.nE() / 2 + 1;
    uint32_t *peel = new uint32_t[hornet.nE() / 2 + 1];
    uint32_t peel_edges = 0;
    uint32_t ne = hornet.nE();
    uint32_t ne_orig = hornet.nE();

    auto pres = vertex_pres;
    auto color = vertex_color;
    auto subg = vertex_subg;
    HornetGraph &h_copy = *h_copy_ptr;
    
    forAllnumV(hornet, [=] __device__ (int i){ pres[i] = 1; } );
    forAllnumV(hornet, [=] __device__ (int i){ subg[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ color[i] = 0; } );

    // HornetGraph &h_copy = *hornet_copy(hornet, h_copy_csr_off,
    //                                   h_copy_csr_edges);

    gpu::memsetZero(hd_data().counter);  // reset counter. 

    forAllEdges(h_copy, ClearHCopy { hd_data }, load_balancing);
    oper_bidirect_batch(h_copy, hd_data().src, hd_data().dst, hornet.nE(), DELETE);

    hornet.reserveBatchOpResource(hornet.nE(),
                                     gpu::batch_property::IN_PLACE | 
                                     gpu::batch_property::REMOVE_BATCH_DUPLICATE |
                                     gpu::batch_property::REMOVE_CROSS_DUPLICATE);

    h_copy.reserveBatchOpResource(hornet.nE(),
                                     gpu::batch_property::IN_PLACE | 
                                     gpu::batch_property::REMOVE_BATCH_DUPLICATE |
                                     gpu::batch_property::REMOVE_CROSS_DUPLICATE);

    uint32_t iter_count = 0; 
    int size = 0;

    Timer<DEVICE> TM;
    TM.start();
    while (peel_edges < ne_orig / 2) {
        uint32_t max_peel = 0;
        ne = ne_orig - 2 * peel_edges;

        if (iter_count % 2) {
            kcores(h_copy, hornet, vqueue, hd_data, peel_vqueue, 
                   load_balancing, &max_peel, vertex_pres, vertex_subg, 
                   // vertex_color, &ne, peel_edges, batch_update);
                   vertex_color, &ne, peel_edges);
            
            forAllVertices(hornet, SetPresent { vertex_pres });
        } else {
            kcores(hornet, h_copy, vqueue, hd_data, peel_vqueue, 
                   load_balancing, &max_peel, vertex_pres, vertex_subg, 
                   // vertex_color, &ne, peel_edges, batch_update);
                   vertex_color, &ne, peel_edges);

            forAllVertices(h_copy, SetPresent { vertex_pres });
        }

        
        std::cout << "max_peel: " << max_peel << "\n";

        cudaMemcpy(&size, hd_data().counter, sizeof(int), 
                   cudaMemcpyDeviceToHost);
        size /= 2;

        if (size > 0) {
            #if 0
            cudaMemcpy(src + peel_edges, hd_data().src, 
                       size * sizeof(vid_t), cudaMemcpyDeviceToHost);

            cudaMemcpy(dst + peel_edges, hd_data().dst, 
                       size * sizeof(vid_t), cudaMemcpyDeviceToHost);
            #endif

            #pragma omp parallel for
            for (uint32_t i = 0; i < size; i++) {
                peel[peel_edges + i] = max_peel;
            }

            peel_edges += size;
        }

        iter_count++;

        if (peel_edges >= len) {
            std::cout << "ooooops" << std::endl;
            std::cout << "peel_edges " << peel_edges << " len " << len << std::endl;
        }
    }
    TM.stop();
    TM.print("KCore");

    cudaMemcpy(src, hd_data().src_tot, 
               peel_edges * sizeof(vid_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(dst, hd_data().dst_tot, 
                peel_edges * sizeof(vid_t), cudaMemcpyDeviceToHost);

    json_dump(src, dst, peel, peel_edges);
}
#endif

#if !defined SHELL && defined NEW_KCORE
void KCore::run() {
    omp_set_num_threads(72);
    vid_t *src     = new vid_t[hornet.nE()];
    vid_t *dst     = new vid_t[hornet.nE()];
    uint32_t len = hornet.nE() / 2 + 1;
    uint32_t *peel = new uint32_t[hornet.nE()];
    uint32_t peel_edges = 0;
    uint32_t ne = hornet.nE();
    std::cout << "ne: " << ne << "\n";

    auto pres = vertex_pres;
    auto deg = vertex_deg;
    auto color = vertex_color;
    
    forAllnumV(hornet, [=] __device__ (int i){ pres[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ deg[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ color[i] = 0; } );

    Timer<DEVICE> TM;
    TM.start();

    // Find vertices of degree 1.
    forAllVertices(hornet, GetDegOne { vqueue, vertex_color });
    vqueue.swap();
    // std::cout << "size: " << vqueue.size() << "\n";
    // Find the edges incident to these vertices.
    gpu::memsetZero(hd_data().counter);  // reset counter. 
    forAllEdges(hornet, vqueue, 
                    DegOneEdges { hd_data, vertex_color }, load_balancing);
    // Mark edges with peel 1.
    int peel_one_count = 0;
    cudaMemcpy(&peel_one_count, hd_data().counter, sizeof(int), cudaMemcpyDeviceToHost);
    #pragma omp parallel for
    for (uint32_t i = 0; i < peel_one_count; i++) {
        peel[i] = 1;
    }

    cudaMemcpy(src, hd_data().src, peel_one_count * sizeof(vid_t), 
                    cudaMemcpyDeviceToHost);
    cudaMemcpy(dst, hd_data().dst, peel_one_count * sizeof(vid_t), 
                    cudaMemcpyDeviceToHost);

    peel_edges = (uint32_t)peel_one_count;

    // Delete peel 1 edges.
    oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, peel_one_count, DELETE);

    // Find all vertices of degree 1.
    while (peel_edges < ne) {
        // std::cout << "peel edges: " << peel_edges << std::endl;
        uint32_t max_peel = 0;
        int batch_size = 0;
        kcores_new(hornet, hd_data, peel_vqueue, active_queue, iter_queue, 
                   load_balancing, vertex_deg, vertex_pres, &max_peel, &batch_size);
        std::cout << "max_peel: " << max_peel << "\n";

        if (batch_size > 0) {
            cudaMemcpy(src + peel_edges, hd_data().src, 
                       batch_size * sizeof(vid_t), cudaMemcpyDeviceToHost);

            cudaMemcpy(dst + peel_edges, hd_data().dst, 
                       batch_size * sizeof(vid_t), cudaMemcpyDeviceToHost);

            #pragma omp parallel for
            for (uint32_t i = 0; i < batch_size; i++) {
                peel[peel_edges + i] = max_peel;
            }

            peel_edges += batch_size;
        }
        oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, batch_size, DELETE);
    }
    TM.stop();
    TM.print("KCore");
    json_dump(src, dst, peel, peel_edges);
}
#endif

void KCore::release() {
    std::cout << "ran3" << std::endl;
}
}
