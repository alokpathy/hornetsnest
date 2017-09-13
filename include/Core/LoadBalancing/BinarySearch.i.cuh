/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "BinarySearchKernel.cuh"
#include <Device/CubWrapper.cuh>    //xlib::CubExclusiveSum
#include <Device/Definition.cuh>    //xlib::SMemPerBlock

namespace load_balacing {

template<typename HornetClass>
BinarySearch::BinarySearch(const HornetClass& hornet) noexcept {
    static_assert(IsHornet<HornetClass>::value,
                 "BinarySearch: paramenter is not an instance of Hornet Class");
    cuMalloc(_d_work, hornet.nV() + 1);
}

inline BinarySearch::~BinarySearch() noexcept {
    cuFree(_d_work);
}

template<typename HornetClass, typename Operator>
void BinarySearch::apply(const HornetClass& hornet,
                         const vid_t*       d_input,
                         int                num_vertices,
                         const Operator&    op) const noexcept {
    static_assert(IsHornet<HornetClass>::value,
                 "BinarySearch: paramenter is not an instance of Hornet Class");
    const auto ITEMS_PER_BLOCK = xlib::SMemPerBlock<BLOCK_SIZE, vid_t>::value;

    kernel::computeWorkKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (d_input, hornet.device_degrees(), num_vertices, _d_work);
    CHECK_CUDA_ERROR

    xlib::CubExclusiveSum<int> prefixsum(_d_work, num_vertices + 1);
    prefixsum.run();

    int total_work;
    cuMemcpyToHostAsync(_d_work + num_vertices, total_work);
    unsigned grid_size = xlib::ceil_div<ITEMS_PER_BLOCK>(total_work);

    if (total_work == 0)
        return;
    kernel::binarySearchKernel<BLOCK_SIZE, ITEMS_PER_BLOCK>
        <<< grid_size, BLOCK_SIZE >>>
        (hornet.device_side(), d_input, _d_work, num_vertices + 1, op);
    CHECK_CUDA_ERROR
}

template<typename HornetClass, typename Operator>
void BinarySearch::apply(const HornetClass& hornet, const Operator& op)
                         const noexcept {
    static_assert(IsHornet<HornetClass>::value,
                 "BinarySearch: paramenter is not an instance of Hornet Class");
    const auto ITEMS_PER_BLOCK = xlib::SMemPerBlock<BLOCK_SIZE, vid_t>::value;

    unsigned grid_size = xlib::ceil_div<ITEMS_PER_BLOCK>(hornet.nE());

    kernel::binarySearchKernel<BLOCK_SIZE, ITEMS_PER_BLOCK>
        <<< grid_size, BLOCK_SIZE >>>
        (hornet.device_side(), hornet.device_csr_offsets(),
         hornet.nV() + 1, op);
    CHECK_CUDA_ERROR
}

/*
template<void (*Operator)(hornet::Edge&, void*)>
void BinarySearch::apply(const hornet::vid_t* d_input, int num_vertices,
                         void* optional_data) noexcept {
    using hornet::vid_t;
    const int ITEMS_PER_BLOCK = xlib::SMemPerBlock<BLOCK_SIZE, vid_t>::value;

    detail::computeWorkKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (d_input, _d_degrees, num_vertices, _d_work);
    CHECK_CUDA_ERROR

    xlib::CubExclusiveSum<int> prefixsum(_d_work, num_vertices + 1);
    prefixsum.run();

    int total_work;
    cuMemcpyToHost(_d_work + num_vertices, total_work);
    unsigned grid_size = xlib::ceil_div<ITEMS_PER_BLOCK>(total_work);

    binarySearchKernel<BLOCK_SIZE, ITEMS_PER_BLOCK, Operator>
        <<< grid_size, BLOCK_SIZE >>>
        (_custinger.device_side(), d_input, _d_work, num_vertices + 1,
         optional_data);
    CHECK_CUDA_ERROR
}*/

} // namespace load_balacing
