#pragma once

#include "HornetAlg.hpp"
#include "Core/HostDeviceVar.cuh"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>

namespace hornets_nest {

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

struct KCoreData {
    vid_t *src;
    vid_t *dst;
    vid_t *src_tot;
    vid_t *dst_tot;
    int   *counter;
    int   *counter_tot;
};

class KCore : public StaticAlgorithm<HornetGraph> {
public:
    KCore(HornetGraph &hornet);
    ~KCore();

    void reset()    override;
    void run()      override;
    void old_run();     
    void release()  override;
    bool validate() override { return true; }
    void set_hcopy(HornetGraph *h_copy);

private:
    HostDeviceVar<KCoreData> hd_data;
    HornetGraph *h_copy_ptr;

    //load_balancing::BinarySearch load_balancing;
    load_balancing::VertexBased1 load_balancing;

    TwoLevelQueue<vid_t> vqueue;
    // TwoLevelQueue<vid_t> src_equeue;
    // TwoLevelQueue<vid_t> dst_equeue;
    TwoLevelQueue<vid_t> peel_vqueue;
    TwoLevelQueue<vid_t> active_queue;
    TwoLevelQueue<vid_t> iter_queue;
    // TwoLevelQueue<vid_t> tot_src_equeue;
    // TwoLevelQueue<vid_t> tot_dst_equeue;

    vid_t *vertex_pres { nullptr };
    vid_t *vertex_color { nullptr };
    vid_t *vertex_subg { nullptr };
    vid_t *h_copy_csr_off { nullptr };
    vid_t *h_copy_csr_edges { nullptr };
    vid_t *vertex_deg { nullptr };

    // HostDeviceVar<KCoreData> hd_data;
    // MultiLevelQueue<vid_t> nodes_removed;
};

}
