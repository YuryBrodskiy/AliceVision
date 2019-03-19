// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
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

class SemiGlobalMatchingVolume
{
public:
    SemiGlobalMatchingVolume(float _volGpuMB, int _volDimX, int _volDimY, int _volDimZ, SemiGlobalMatchingParams* _sp);
    ~SemiGlobalMatchingVolume(void);

    void copyVolume(const StaticVector<int>* volume);
    void copyVolume(const StaticVector<unsigned char>* volume, int zFrom, int nZSteps);

    void SemiGlobalMatchingVolume::copyVolume(const thrust::host_vector<unsigned char, thrust::cuda::experimental::pinned_allocator<unsigned char>>* volume, int zFrom, int nZSteps);

    void addVolumeMin(const StaticVector<unsigned char>* volume, int zFrom, int nZSteps);
    void addVolumeSecondMin(const StaticVector<unsigned char>* volume, int zFrom, int nZSteps);

	void SemiGlobalMatchingVolume::addVolumeSecondMin(const thrust::host_vector<unsigned char, thrust::cuda::experimental::pinned_allocator<unsigned char>>* volume, int zFrom, int nZSteps);

    void addVolumeAvg(int n, const StaticVector<unsigned char>* volume, int zFrom, int nZSteps);

    void cloneVolumeStepZ();
    void cloneVolumeSecondStepZ(int rc);

	void SemiGlobalMatchingVolume::cloneVolumeSecondStepZPinnedMemory(int rc);

    void SGMoptimizeVolumeStepZ(int rc, int volStepXY, int volLUX, int volLUY, int scale);

	void SemiGlobalMatchingVolume::SGMoptimizeVolumeStepZPinnedMemory(int rc, int volStepXY, int volLUX, int volLUY, int scale);


    StaticVector<IdValue>* getOrigVolumeBestIdValFromVolumeStepZ(int zborder);


	StaticVector<IdValue>* SemiGlobalMatchingVolume::getOrigVolumeBestIdValFromVolumeStepZPinnedMemory(int zborder);

	void WriteVolumeStepBestToFilePinnedMemory(int rc);

	void WriteVolumeStepBestToFile(int rc);


	void SemiGlobalMatchingVolume::WriteOptimizeVolumeToFilePinned(int rc, thrust::host_vector<unsigned char, thrust::cuda::experimental::pinned_allocator<unsigned char>>* _volumeStepZPinnedMemory);

	void SemiGlobalMatchingVolume::WriteOptimizeVolumeToFile(int rc, StaticVector<unsigned char>* _volumeStepZ);

	StaticVector<IdValue>* SemiGlobalMatchingVolume::getOrigVolumeBestIdValFromVolumeStepZPinned(int zborder);





private:
    SemiGlobalMatchingParams* sp;

    float volGpuMB;
    int volDimX;
    int volDimY;
    int volDimZ;
    int volStepZ;

    /// Volume containing the second best value accross multiple input volumes
    StaticVector<unsigned char>* _volumeSecondBest;


	thrust::host_vector<unsigned char, thrust::cuda::experimental::pinned_allocator<unsigned char>>* _volumeSecondBestPinnedMmeory;


    /// Volume containing the best value accross multiple input volumes
    StaticVector<unsigned char>* _volume;

	/// Volume containing the best value accros multiple input volumes but the vector has been allocated in pinned memory
	thrust::host_vector<unsigned char, thrust::cuda::experimental::pinned_allocator<unsigned char>>* volumePinnedMemory;

    /// The similarity volume after Z reduction. Volume dimension is (X, Y, Z/step).
    StaticVector<unsigned char>* _volumeStepZ;


	thrust::host_vector<unsigned char, thrust::cuda::experimental::pinned_allocator<unsigned char>>* _volumeStepZPinnedMemory;


    /// Volume with the index of the original plane. Volume dimension (X, Y, Z/step).
    StaticVector<int>* _volumeBestZ;


	thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int>>* _volumeBestZPinnedMemory;
};

} // namespace depthMap
} // namespace aliceVision
