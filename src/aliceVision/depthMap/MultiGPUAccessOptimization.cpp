// This file is part of the extention to AliceVision project.
// Copyright (c) 2018 EIVA a/s
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
#include "MultiGPUAccessOptimization.hpp"
#include "aliceVision/alicevision_omp.hpp"

namespace aliceVision
{
namespace depthMap
{

void doOnGPUs(mvsUtils::MultiViewParams* mp, mvsUtils::PreMatchCams* pc, const StaticVector<int>& cams, GPUJob gpujob)
{
    int num_gpus = listCUDADevices(true);
    int num_cpu_threads = omp_get_num_procs();
    ALICEVISION_LOG_INFO("Number of GPU devices: " << num_gpus << ", number of CPU threads: " << num_cpu_threads);
    int numthreads = std::min(num_gpus, num_cpu_threads);

    int num_gpus_to_use = mp->_ini.get<int>("refineRc.num_gpus_to_use", 1);
    if(num_gpus_to_use > 0)
    {
        numthreads = num_gpus_to_use;
    }

    if(numthreads == 1)
    {
        gpujob(mp->CUDADeviceNo, mp, pc, cams);
    }
    else
    {
        omp_set_num_threads(numthreads); // create as many CPU threads as there are CUDA devices
#pragma omp parallel
        {
            int cpu_thread_id = omp_get_thread_num();
            int CUDADeviceNo = cpu_thread_id % numthreads;
            ALICEVISION_LOG_INFO("CPU thread " << cpu_thread_id << " (of " << numthreads
                                               << ") uses CUDA device: " << CUDADeviceNo);

            int rcFrom = CUDADeviceNo * (cams.size() / numthreads);
            int rcTo = (CUDADeviceNo + 1) * (cams.size() / numthreads);
            if(CUDADeviceNo == numthreads - 1)
            {
                rcTo = cams.size();
            }
            StaticVector<int> subcams;
            subcams.reserve(cams.size());
            for(int rc = rcFrom; rc < rcTo; rc++)
            {
                subcams.push_back(cams[rc]);
            }
            gpujob(cpu_thread_id, mp, pc, subcams);
        }
    }
}

} // namespace depthMap
} // namespace aliceVision
