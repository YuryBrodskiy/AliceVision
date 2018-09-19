// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/system/cmdline.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/Timer.hpp>
#include <aliceVision/mvsData/StaticVector.hpp>
#include <aliceVision/mvsUtils/common.hpp>
#include <aliceVision/mvsUtils/MultiViewParams.hpp>
#include <aliceVision/mvsUtils/PreMatchCams.hpp>
#include <aliceVision/depthMap/RefineRc.hpp>
#include <aliceVision/depthMap/SemiGlobalMatchingRc.hpp>
#include <aliceVision/system/gpu.hpp>
#include <aliceVision/depthMap/MultiGPUAccessOptimization.hpp>
#include <aliceVision/imageIO/image.hpp>
#include <aliceVision/mvsUtils/fileIO.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

using namespace aliceVision;

namespace bfs = boost::filesystem;
namespace po = boost::program_options;


int main(int argc, char* argv[])
{
    system::Timer timer;

    std::string verboseLevel = system::EVerboseLevel_enumToString(system::Logger::getDefaultVerboseLevel());
    std::string iniFilepath;
    std::string outputFolder;

    int rangeStart = -1;
    int rangeSize = -1;

    // image downscale factor during process
    int downscale = 2;

    // semiGlobalMatching
    int sgmMaxTCams = 10;
    int sgmWSH = 4;
    double sgmGammaC = 5.5;
    double sgmGammaP = 8.0;

    // refineRc
    int refineNSamplesHalf = 150;
    int refineNDepthsToRefine = 31;
    int refineNiters = 100;
    int refineWSH = 3;
    int refineMaxTCams = 6;
    double refineSigma = 15.0;
    double refineGammaC = 15.5;
    double refineGammaP = 8.0;
    bool refineUseTcOrRcPixSize = false;

    po::options_description allParams("AliceVision depthMapEstimation\n"
                                      "Estimate depth map for each input image");

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("ini", po::value<std::string>(&iniFilepath)->required(),
            "Configuration file (mvs.ini).")
        ("output,o", po::value<std::string>(&outputFolder)->required(),
            "Output folder for generated depth maps.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("rangeStart", po::value<int>(&rangeStart)->default_value(rangeStart),
            "Compute a sub-range of images from index rangeStart to rangeStart+rangeSize.")
        ("rangeSize", po::value<int>(&rangeSize)->default_value(rangeSize),
            "Compute a sub-range of N images (N=rangeSize).")
        ("downscale", po::value<int>(&downscale)->default_value(downscale),
            "Image downscale factor.")
        ("sgmMaxTCams", po::value<int>(&sgmMaxTCams)->default_value(sgmMaxTCams),
            "Semi Global Matching: Number of neighbour cameras.")
        ("sgmWSH", po::value<int>(&sgmWSH)->default_value(sgmWSH),
            "Semi Global Matching: Size of the patch used to compute the similarity.")
        ("sgmGammaC", po::value<double>(&sgmGammaC)->default_value(sgmGammaC),
            "Semi Global Matching: GammaC threshold.")
        ("sgmGammaP", po::value<double>(&sgmGammaP)->default_value(sgmGammaP),
            "Semi Global Matching: GammaP threshold.")
        ("refineNSamplesHalf", po::value<int>(&refineNSamplesHalf)->default_value(refineNSamplesHalf),
            "Refine: Number of samples.")
        ("refineNDepthsToRefine", po::value<int>(&refineNDepthsToRefine)->default_value(refineNDepthsToRefine),
            "Refine: Number of depths.")
        ("refineNiters", po::value<int>(&refineNiters)->default_value(refineNiters),
            "Refine: Number of iterations.")
        ("refineWSH", po::value<int>(&refineWSH)->default_value(refineWSH),
            "Refine: Size of the patch used to compute the similarity.")
        ("refineMaxTCams", po::value<int>(&refineMaxTCams)->default_value(refineMaxTCams),
            "Refine: Number of neighbour cameras.")
        ("refineSigma", po::value<double>(&refineSigma)->default_value(refineSigma),
            "Refine: Sigma threshold.")
        ("refineGammaC", po::value<double>(&refineGammaC)->default_value(refineGammaC),
            "Refine: GammaC threshold.")
        ("refineGammaP", po::value<double>(&refineGammaP)->default_value(refineGammaP),
            "Refine: GammaP threshold.")
        ("refineUseTcOrRcPixSize", po::value<bool>(&refineUseTcOrRcPixSize)->default_value(refineUseTcOrRcPixSize),
            "Refine: Use current camera pixel size or minimum pixel size of neighbour cameras.");

    po::options_description logParams("Log parameters");
    logParams.add_options()
      ("verboseLevel,v", po::value<std::string>(&verboseLevel)->default_value(verboseLevel),
        "verbosity level (fatal, error, warning, info, debug, trace).");

    allParams.add(requiredParams).add(optionalParams).add(logParams);

    po::variables_map vm;

    try
    {
      po::store(po::parse_command_line(argc, argv, allParams), vm);

      if(vm.count("help") || (argc == 1))
      {
        ALICEVISION_COUT(allParams);
        return EXIT_SUCCESS;
      }

      po::notify(vm);
    }
    catch(boost::program_options::required_option& e)
    {
      ALICEVISION_CERR("ERROR: " << e.what() << std::endl);
      ALICEVISION_COUT("Usage:\n\n" << allParams);
      return EXIT_FAILURE;
    }
    catch(boost::program_options::error& e)
    {
      ALICEVISION_CERR("ERROR: " << e.what() << std::endl);
      ALICEVISION_COUT("Usage:\n\n" << allParams);
      return EXIT_FAILURE;
    }

    ALICEVISION_COUT("Program called with the following parameters:");
    ALICEVISION_COUT(vm);

    // set verbose level
    system::Logger::get()->setLogLevel(verboseLevel);

    // print GPU Information
    ALICEVISION_LOG_INFO(system::gpuInformationCUDA());

    // check if the gpu suppport CUDA compute capability 2.0
    if(!system::gpuSupportCUDA(2,0))
    {
      ALICEVISION_LOG_ERROR("This program needs a CUDA-Enabled GPU (with at least compute capablility 2.0).");
      return EXIT_FAILURE;
    }

    // check if the scale is correct
    if(downscale < 1)
    {
      ALICEVISION_LOG_ERROR("Invalid value for downscale parameter. Should be at least 1.");
      return EXIT_FAILURE;
    }

    // .ini and files parsing
    mvsUtils::MultiViewParams mp(iniFilepath, outputFolder, "", false, downscale);

    // set params in bpt

    // semiGlobalMatching
    mp._ini.put("semiGlobalMatching.maxTCams", sgmMaxTCams);
    mp._ini.put("semiGlobalMatching.wsh", sgmWSH);
    mp._ini.put("semiGlobalMatching.gammaC", sgmGammaC);
    mp._ini.put("semiGlobalMatching.gammaP", sgmGammaP);

    // refineRc
    mp._ini.put("refineRc.nSamplesHalf", refineNSamplesHalf);
    mp._ini.put("refineRc.ndepthsToRefine", refineNDepthsToRefine);
    mp._ini.put("refineRc.niters", refineNiters);
    mp._ini.put("refineRc.wsh", refineWSH);
    mp._ini.put("refineRc.maxTCams", refineMaxTCams);
    mp._ini.put("refineRc.sigma", refineSigma);
    mp._ini.put("refineRc.gammaC", refineGammaC);
    mp._ini.put("refineRc.gammaP", refineGammaP);
    mp._ini.put("refineRc.useTcOrRcPixSize", refineUseTcOrRcPixSize);

    mvsUtils::PreMatchCams pc(&mp);

    StaticVector<int> cams;
    cams.reserve(mp.ncams);
    if(rangeSize == -1)
    {
        for(int rc = 0; rc < mp.ncams; rc++) // process all cameras
            cams.push_back(rc);
    }
    else
    {
        if(rangeStart < 0)
        {
            ALICEVISION_LOG_ERROR("invalid subrange of cameras to process.");
            return EXIT_FAILURE;
        }
        for(int rc = rangeStart; rc < std::min(rangeStart + rangeSize, mp.ncams); ++rc)
            cams.push_back(rc);
        if(cams.empty())
        {
            ALICEVISION_LOG_INFO("No camera to process.");
            return EXIT_SUCCESS;
        }
    }

    ALICEVISION_LOG_INFO("Create depth maps.");

    {
        depthMap::doOnGPUs(&mp, &pc, cams, depthMap::processImageStream);
    }

    ALICEVISION_LOG_INFO("Task done in (s): " + std::to_string(timer.elapsed()));



    return EXIT_SUCCESS;
}
