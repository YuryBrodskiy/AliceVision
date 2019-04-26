// This file is part of the extention to AliceVision project.
// Copyright (c) 2018 EIVA a/s
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
#include "MultiGPUAccessOptimization.hpp"
#include "aliceVision/alicevision_omp.hpp"
#include <aliceVision/mvsUtils/common.hpp>
#include <aliceVision/depthMap/SemiGlobalMatchingRc.hpp>
#include <aliceVision/depthMap/RefineRc.hpp>
#include <aliceVision/mvsUtils/fileIO.hpp>
#include <aliceVision/imageIO/image.hpp>

namespace aliceVision
{
namespace depthMap
{
//void savePointCloudXYZ(mvsUtils::ImagesCache& imageCache, int rc, mvsUtils::MultiViewParams* mp)
//{
//    imageCache.refreshData(rc);
//    std::string xyzFileName = mv_getFileNamePrefix(mp->getDepthMapFolder(), mp, rc) + "PCL.xyz";
//    FILE* f = fopen(xyzFileName.c_str(), "w");
//    int w = mp->getWidth(rc);
//    int h = mp->getHeight(rc);
//    StaticVector<float> depthMap;
//
//    {
//        int width, height;
//        imageIO::readImage(mv_getFileName(mp, rc, mvsUtils::EFileType::depthMap, 1), width, height, depthMap.getDataWritable());
//		imageIO::transposeImage(width, height, depthMap.getDataWritable());
//    }
//
//    if((depthMap.empty()) || (depthMap.size() != w * h))
//    {
//        std::stringstream s;
//        s << "filterGroupsRC: bad image dimension for camera: " << mp->getViewId(rc) << "\n";
//        s << "depthMap size: " << depthMap.size() << ", width: " << w << ", height: " << h;
//        throw std::runtime_error(s.str());
//    }
//    else
//    {
//        for(int i = 0; i < sizeOfStaticVector<float>(&depthMap); i++)
//        {
//            int x = i / h;
//            int y = i % h;
//            float depth = depthMap[i];
//            if(depth > 0.0f)
//            {
//                Point3d p = mp->CArr[rc] + (mp->iCamArr[rc] * Point2d((float)x, (float)y)).normalize() * depth;
//                Point2d pixRC;
//                mp->getPixelFor3DPoint(&pixRC, p, rc);
//                if(mp->isPixelInImage(pixRC, rc))
//                {
//                    Color color = imageCache.getPixelValueInterpolated(&pixRC, rc);
//                    fprintf(f, "%f %f %f %f %f %f\n", p.x, p.y, p.z, color.r * 255, color.g * 255, color.b * 255);
//                }
//            }
//        }
//    }
//    fclose(f);
//}


//void processImageStream(int CUDADeviceNo, mvsUtils::MultiViewParams* mp, mvsUtils::PreMatchCams* pc, const StaticVector<int>& cams)
//{
//    aliceVision::mvsUtils::SGMParams sgm(mp);
//    const int bandType = 0;
//    // load images from files into RAM
//    mvsUtils::ImagesCache ic(mp, bandType, true);
//    // load stuff on GPU memory and creates multi-level images and computes gradients
//    PlaneSweepingCuda cps(CUDADeviceNo, &ic, mp, pc, sgm.scale, cams); // ToDo add cameras to load
//    // init plane sweeping parameters
//    SemiGlobalMatchingParams sp(mp, pc, &cps);
//
//    //////////////////////////////////////////////////////////////////////////////////////////
//    StaticVector<int> cams_used_times;
//    cams_used_times.resize_with(mp->ncams, 0);
//    int max_use_cam = mp->_ini.get<int>("semiGlobalMatching.maxUse", 1);
//    int nnearestcams = mp->_ini.get<int>("semiGlobalMatching.maxTCams", 10);
//    for(const int rc : cams)
//    {
//		// can be used to speed up the process substantially
//        //if(cams_used_times[rc] < max_use_cam)
//        {
////             StaticVector<int> ncams = pc->findNearestCamsFromSeeds(rc, nnearestcams);
////             for(int lc : ncams)
////             {
////                 cams_used_times[lc] += 1;
////             }
//            std::string depthMapFilepath = sp.getSGM_idDepthMapFileName(mp->getViewId(rc), sgm.scale, sgm.step);
//            if(!mvsUtils::FileExists(depthMapFilepath))
//            {
//                ALICEVISION_LOG_INFO("Compute depth map: " << depthMapFilepath);
//                SemiGlobalMatchingRc psgr(true, rc, sgm.scale, sgm.step, &sp);
//                psgr.sgmrc();
//            }
//            else
//            {
//                ALICEVISION_LOG_INFO("Depth map already computed: " << depthMapFilepath);
//            }
//            std::string depthMapRefinedFilePath =
//                sp.getREFINE_opt_simMapFileName(mp->getViewId(rc), sgm.scale, sgm.step);
//            if(!mvsUtils::FileExists(depthMapRefinedFilePath))
//            {
//                ALICEVISION_LOG_INFO("Refine depth map: " << depthMapRefinedFilePath);
//                RefineRc rrc(rc, sgm.scale, sgm.step, &sp);
//                rrc.refinercCUDA();
//            }
//            else
//            {
//                ALICEVISION_LOG_INFO("Depth map already computed: " << depthMapRefinedFilePath);
//            }
//
//            depthMap::savePointCloudXYZ(ic, rc, mp);
//        }
//    }
//}

//void doOnGPUs(mvsUtils::MultiViewParams* mp, mvsUtils::PreMatchCams* pc, const StaticVector<int>& cams, GPUJob gpujob)
//{
//    int num_gpus = listCUDADevices(true);
//    int num_cpu_threads = omp_get_num_procs();
//    ALICEVISION_LOG_INFO("Number of GPU devices: " << num_gpus << ", number of CPU threads: " << num_cpu_threads);
//    int numthreads = std::min(num_gpus, num_cpu_threads);
//
//    int num_gpus_to_use = mp->_ini.get<int>("refineRc.num_gpus_to_use", 1);
//    if(num_gpus_to_use > 0)
//    {
//        numthreads = num_gpus_to_use;
//    }
//
//    if(numthreads == 1)
//    {
//        gpujob(mp->CUDADeviceNo, mp, pc, cams);
//    }
//    else
//    {
//        omp_set_num_threads(numthreads); // create as many CPU threads as there are CUDA devices
//#pragma omp parallel
//        {
//            int cpu_thread_id = omp_get_thread_num();
//            int CUDADeviceNo = cpu_thread_id % numthreads;
//            ALICEVISION_LOG_INFO("CPU thread " << cpu_thread_id << " (of " << numthreads
//                                               << ") uses CUDA device: " << CUDADeviceNo);
//
//            int rcFrom = CUDADeviceNo * (cams.size() / numthreads);
//            int rcTo = (CUDADeviceNo + 1) * (cams.size() / numthreads);
//            if(CUDADeviceNo == numthreads - 1)
//            {
//                rcTo = cams.size();
//            }
//            StaticVector<int> subcams;
//            subcams.reserve(cams.size());
//            for(int rc = rcFrom; rc < rcTo; rc++)
//            {
//                subcams.push_back(cams[rc]);
//            }
//            gpujob(cpu_thread_id, mp, pc, subcams);
//        }
//    }
//}

} // namespace depthMap
} // namespace aliceVision
