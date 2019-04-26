//
//  Task sink in C++
//  Binds PULL socket to tcp://localhost:5558
//  Collects results from workers via that socket
//
//  Olivier Chamoux <olivier.chamoux@fr.thalesgroup.com>
//
#include <zmq.hpp>
#include <time.h>
#include <aliceVision/system/Timer.hpp>
#include <iostream>

int main(int argc, char* argv[])
{
    //  Prepare our context and socket
    zmq::context_t context(1);
    zmq::socket_t receiver(context, ZMQ_PULL);
    receiver.bind("tcp://*:5558");

    //  Wait for start of batch
    zmq::message_t message;
    receiver.recv(&message);


    aliceVision::system::Timer timer;

    //  Process 100 confirmations
    int task_nbr;
    int total_msec = 0; //  Total calculated cost in msecs
    for(task_nbr = 0; task_nbr < 100; task_nbr++)
    {

        receiver.recv(&message);
        if((task_nbr / 10) * 10 == task_nbr)
            std::cout << ":" << std::flush;
        else
            std::cout << "." << std::flush;
    }
    //  Calculate and report duration of batch

    std::cout << "\nTotal elapsed time: " << std::to_string(timer.elapsed()) << " msec\n" << std::endl;
    return 0;
}


//// This file is part of the AliceVision project.
//// Copyright (c) 2017 AliceVision contributors.
//// This Source Code Form is subject to the terms of the Mozilla Public License,
//// v. 2.0. If a copy of the MPL was not distributed with this file,
//// You can obtain one at https://mozilla.org/MPL/2.0/.
//
//#include <aliceVision/system/cmdline.hpp>
//#include <aliceVision/system/Logger.hpp>
//#include <aliceVision/system/Timer.hpp>
//#include <aliceVision/mvsData/StaticVector.hpp>
//#include <aliceVision/mvsUtils/common.hpp>
//#include <aliceVision/mvsUtils/MultiViewParams.hpp>
//#include <aliceVision/mvsUtils/PreMatchCams.hpp>
//#include <aliceVision/fuseCut/Fuser.hpp>
//#include <aliceVision/imageIO/image.hpp>
//#include <aliceVision/mvsUtils/fileIO.hpp>
//#include <aliceVision/mvsUtils/ImagesCache.hpp>
//
//#include <boost/program_options.hpp>
//#include <boost/filesystem.hpp>
//
//// These constants define the current software version.
//// They must be updated when the command line is changed.
//#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
//#define ALICEVISION_SOFTWARE_VERSION_MINOR 0
//
//using namespace aliceVision;
//
//namespace bfs = boost::filesystem;
//namespace po = boost::program_options;
//
//int main(int argc, char* argv[])
//{
//    system::Timer timer;
//
//    std::string verboseLevel = system::EVerboseLevel_enumToString(system::Logger::getDefaultVerboseLevel());
//    std::string iniFilepath;
//    std::string depthMapFolder;
//    std::string outputFolder;
//    int rangeStart = -1;
//    int rangeSize = -1;
//
//    int minNumOfConsistensCams = 3;
//    int minNumOfConsistensCamsWithLowSimilarity = 4;
//    int pixSizeBall = 0;
//    int pixSizeBallWithLowSimilarity = 0;
//    int nNearestCams = 10;
//
//    po::options_description allParams("AliceVision depthMapFiltering\n"
//                                      "Filter depth map to remove values that are not consistent with other depth maps");
//
//    po::options_description requiredParams("Required parameters");
//    requiredParams.add_options()
//        ("ini", po::value<std::string>(&iniFilepath)->required(),
//            "Configuration file (mvs.ini).")
//        ("depthMapFolder", po::value<std::string>(&depthMapFolder)->required(),
//            "Input depth map folder.")
//        ("output,o", po::value<std::string>(&outputFolder)->required(),
//            "Output folder for filtered depth maps.");
//
//    po::options_description optionalParams("Optional parameters");
//    optionalParams.add_options()
//        ("rangeStart", po::value<int>(&rangeStart)->default_value(rangeStart),
//            "Compute only a sub-range of images from index rangeStart to rangeStart+rangeSize.")
//        ("rangeSize", po::value<int>(&rangeSize)->default_value(rangeSize),
//            "Compute only a sub-range of N images (N=rangeSize).")
//        ("minNumOfConsistensCams", po::value<int>(&minNumOfConsistensCams)->default_value(minNumOfConsistensCams),
//            "Minimal number of consistent cameras to consider the pixel.")
//        ("minNumOfConsistensCamsWithLowSimilarity", po::value<int>(&minNumOfConsistensCamsWithLowSimilarity)->default_value(minNumOfConsistensCamsWithLowSimilarity),
//            "Minimal number of consistent cameras to consider the pixel when the similarity is weak or ambiguous.")
//        ("pixSizeBall", po::value<int>(&pixSizeBall)->default_value(pixSizeBall),
//            "Filter ball size (in px).")
//        ("pixSizeBallWithLowSimilarity", po::value<int>(&pixSizeBallWithLowSimilarity)->default_value(pixSizeBallWithLowSimilarity),
//            "Filter ball size (in px) when the similarity is weak or ambiguous.")
//        ("nNearestCams", po::value<int>(&nNearestCams)->default_value(nNearestCams),
//            "Number of nearest cameras.");
//
//    po::options_description logParams("Log parameters");
//    logParams.add_options()
//      ("verboseLevel,v", po::value<std::string>(&verboseLevel)->default_value(verboseLevel),
//        "verbosity level (fatal, error, warning, info, debug, trace).");
//
//    allParams.add(requiredParams).add(optionalParams).add(logParams);
//
//    po::variables_map vm;
//
//    try
//    {
//      po::store(po::parse_command_line(argc, argv, allParams), vm);
//
//      if(vm.count("help") || (argc == 1))
//      {
//        ALICEVISION_COUT(allParams);
//        return EXIT_SUCCESS;
//      }
//
//      po::notify(vm);
//    }
//    catch(boost::program_options::required_option& e)
//    {
//      ALICEVISION_CERR("ERROR: " << e.what() << std::endl);
//      ALICEVISION_COUT("Usage:\n\n" << allParams);
//      return EXIT_FAILURE;
//    }
//    catch(boost::program_options::error& e)
//    {
//      ALICEVISION_CERR("ERROR: " << e.what() << std::endl);
//      ALICEVISION_COUT("Usage:\n\n" << allParams);
//      return EXIT_FAILURE;
//    }
//
//    ALICEVISION_COUT("Program called with the following parameters:");
//    ALICEVISION_COUT(vm);
//
//    // set verbose level
//    system::Logger::get()->setLogLevel(verboseLevel);
//
//    // .ini and files parsing
//    mvsUtils::MultiViewParams mp(iniFilepath, depthMapFolder, outputFolder, true);
//    mvsUtils::PreMatchCams pc(&mp);
//
//    StaticVector<int> cams;
//    cams.reserve(mp.ncams);
//
//    if(rangeSize == -1)
//    {
//        for(int rc = 0; rc < mp.ncams; rc++) // process all cameras
//            cams.push_back(rc);
//    }
//    else
//    {
//        if(rangeStart < 0)
//        {
//            ALICEVISION_LOG_ERROR("invalid subrange of cameras to process.");
//            return EXIT_FAILURE;
//        }
//        for(int rc = rangeStart; rc < std::min(rangeStart + rangeSize, mp.ncams); ++rc)
//            cams.push_back(rc);
//        if(cams.empty())
//        {
//            ALICEVISION_LOG_INFO("No camera to process.");
//            return EXIT_SUCCESS;
//        }
//    }
//
//    ALICEVISION_LOG_INFO("Filter depth maps.");
//
//    {
//        fuseCut::Fuser fs(&mp, &pc);
//        fs.filterGroups(cams, pixSizeBall, pixSizeBallWithLowSimilarity, nNearestCams);
//        fs.filterDepthMaps(cams, minNumOfConsistensCams, minNumOfConsistensCamsWithLowSimilarity);
//    }
//
//    ALICEVISION_LOG_INFO("Task done in (s): " + std::to_string(timer.elapsed()));
//
//    ALICEVISION_LOG_INFO("Make Point Clouds");
//    system::Timer timer_PCL;
//    const int bandType = 0;
//    // .ini and files parsing
//    mvsUtils::MultiViewParams mp1(iniFilepath, outputFolder, "", false, 4);
//    mvsUtils::ImagesCache imageCache(&mp1, bandType, true);
//    for(int rc : cams)
//    {
//        imageCache.refreshData(rc);
//        std::string xyzFileName = mv_getFileNamePrefix(mp.getDepthMapFilterFolder(), &mp, rc) + "PCL.xyz";
//        FILE* f = fopen(xyzFileName.c_str(), "w");
//        int w = mp.getWidth(rc);
//        int h = mp.getHeight(rc);
//        StaticVector<float> depthMap;
//
//        {
//            int width, height;
//            imageIO::readImage(mv_getFileName(&mp, rc, mvsUtils::EFileType::depthMap, 0), width, height,
//                               depthMap.getDataWritable());
//
//            imageIO::transposeImage(width, height, depthMap.getDataWritable());
//        }
//
//        if((depthMap.empty()) || (depthMap.size() != w * h))
//        {
//            std::stringstream s;
//            s << "filterGroupsRC: bad image dimension for camera: " << mp.getViewId(rc) << "\n";
//            s << "depthMap size: " << depthMap.size() << ", width: " << w << ", height: " << h;
//            throw std::runtime_error(s.str());
//        }
//        else
//        {
//            for(int i = 0; i < sizeOfStaticVector<float>(&depthMap); i++)
//            {
//                int x = i / h ;
//                int y = i % h ;
//                float depth = depthMap[i];
//                if(depth > 0.0f) 
//                {
//                    //@Yury Raw point cloud ???
//                    Point3d p = mp.CArr[rc] + (mp.iCamArr[rc] * Point2d((float)x, (float)y)).normalize()*depth;
//                    Point2d pixRC;
//                    mp.getPixelFor3DPoint(&pixRC, p, rc);
//                    if(!mp.isPixelInImage(pixRC, rc))
//                    {
//                    }
//                    else
//                    {
//                        Color color = imageCache.getPixelValueInterpolated(&pixRC, rc);
//                        fprintf(f, "%f %f %f %f %f %f\n", p.x, p.y, p.z, color.r * 255, color.g * 255, color.b * 255);
//                    }
//                }
//            }
//        }
//        fclose(f);
//    }
//
//    ALICEVISION_LOG_INFO("Save points to xyz done in (s): " + std::to_string(timer.elapsed()));
//    return EXIT_SUCCESS;
//}
