#pragma once

#include "HornetAlg.hpp"
#include "Core/HostDeviceVar.cuh"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>

#include <stdint.h>

#define NUM_PARTS 1

namespace hornets_nest {

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

struct MsgData {
    float val;
    vid_t dst;
};

class PCPM_PR : public StaticAlgorithm<HornetGraph> {
public:
    PCPM_PR(HornetGraph& hornet);
    ~PCPM_PR();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

private:
    // HostDeviceVar<KCoreData> hd_data;

    //load_balancing::BinarySearch load_balancing;

    vid_t *hornet_csr_off[NUM_PARTS];
    vid_t *hornet_csr_edges[NUM_PARTS];

    TwoLevelQueue<vid_t> vqueue;
    TwoLevelQueue<vid_t> src_equeue;
    TwoLevelQueue<vid_t> dst_equeue;

    float *pr;

    // HostDeviceVar<KCoreData> hd_data;
    // MultiLevelQueue<vid_t> nodes_removed;

    load_balancing::VertexBased1 load_balancing;
};

}
