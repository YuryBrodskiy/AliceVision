// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/mvsData/StaticVector.hpp>
#include <aliceVision/mvsData/Voxel.hpp>
#include <aliceVision/depthMap/SemiGlobalMatchingParams.hpp>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace aliceVision
{
namespace depthMap
{

class SemiGlobalMatchingRcTc
{
public:
    SemiGlobalMatchingRcTc(StaticVector<float>* _rcTcDepths, int _rc, int _tc, int _scale, int _step,
                           SemiGlobalMatchingParams* _sp, StaticVectorBool* _rcSilhoueteMap = NULL);
    ~SemiGlobalMatchingRcTc(void);

    StaticVector<unsigned char>* computeDepthSimMapVolume(float& volumeMBinGPUMem, int wsh, float gammaC, float gammaP);

    thrust::host_vector<unsigned char, thrust::cuda::experimental::pinned_allocator<unsigned char>>*
    SemiGlobalMatchingRcTc::computeDepthSimMapVolumeMemoryPinned(float& volumeMBinGPUMem, int wsh, float gammaC,
                                                                 float gammaP, cudaStream_t &stream);

private:
    StaticVector<Voxel>* getPixels();

    SemiGlobalMatchingParams* sp;

    int rc, tc, scale, step;
    StaticVector<float>* rcTcDepths;
    float epipShift;
    int w, h;
    StaticVectorBool* rcSilhoueteMap;
};

} // namespace depthMap
} // namespace aliceVision
