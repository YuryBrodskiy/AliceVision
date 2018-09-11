// This file is part of the extention to AliceVision project.
// Copyright (c) 2018 EIVA a/s
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/mvsData/StaticVector.hpp>
#include <aliceVision/depthMap/SemiGlobalMatchingParams.hpp>
namespace aliceVision
{
namespace depthMap
{
typedef void (*GPUJob)(int CUDADeviceNo, mvsUtils::MultiViewParams* mp, mvsUtils::PreMatchCams* pc,
                       const StaticVector<int>& cams);


void doOnGPUs(mvsUtils::MultiViewParams* mp, mvsUtils::PreMatchCams* pc, const StaticVector<int>& cams, GPUJob gpujob);

} // namespace depthMap
} // namespace aliceVision
