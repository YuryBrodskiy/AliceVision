// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/system/cmdline.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/Timer.hpp>
#include <aliceVision/mvsData/Point3d.hpp>
#include <aliceVision/mvsData/StaticVector.hpp>
#include <aliceVision/mvsUtils/common.hpp>
#include <aliceVision/mvsUtils/MultiViewParams.hpp>
#include <aliceVision/mvsUtils/PreMatchCams.hpp>
#include <aliceVision/mesh/meshPostProcessing.hpp>
#include <aliceVision/fuseCut/LargeScale.hpp>
#include <aliceVision/fuseCut/ReconstructionPlan.hpp>
#include <aliceVision/fuseCut/DelaunayGraphCut.hpp>
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

enum EPartitioningMode
{
    ePartitioningUndefined = 0,
    ePartitioningSingleBlock = 1,
    ePartitioningAuto = 2,
};

EPartitioningMode EPartitioning_stringToEnum(const std::string& s)
{
    if(s == "singleBlock")
        return ePartitioningSingleBlock;
    if(s == "auto")
        return ePartitioningAuto;
    return ePartitioningUndefined;
}

inline std::istream& operator>>(std::istream& in, EPartitioningMode& out_mode)
{
    std::string s;
    in >> s;
    out_mode = EPartitioning_stringToEnum(s);
    return in;
}

enum ERepartitionMode
{
    eRepartitionUndefined = 0,
    eRepartitionMultiResolution = 1,
    eRepartitionRegularGrid = 2,
};

ERepartitionMode ERepartitionMode_stringToEnum(const std::string& s)
{
    if(s == "multiResolution")
        return eRepartitionMultiResolution;
    if(s == "regularGrid")
        return eRepartitionRegularGrid;
    return eRepartitionUndefined;
}

inline std::istream& operator>>(std::istream& in, ERepartitionMode& out_mode)
{
    std::string s;
    in >> s;
    out_mode = ERepartitionMode_stringToEnum(s);
    return in;
}


int main(int argc, char* argv[])
{
    system::Timer timer;

    std::string verboseLevel = system::EVerboseLevel_enumToString(system::Logger::getDefaultVerboseLevel());
    std::string iniFilepath;
    std::string outputMesh;
    std::string depthMapFolder;
    std::string depthMapFilterFolder;

    EPartitioningMode partitioningMode = ePartitioningSingleBlock;
    ERepartitionMode repartitionMode = eRepartitionMultiResolution;
    po::options_description inputParams;
    int maxPtsPerVoxel = 6000000;

    fuseCut::FuseParams fuseParams;

    po::options_description allParams("AliceVision meshing");

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("ini", po::value<std::string>(&iniFilepath)->required(),
            "Configuration file (mvs.ini).")
        ("depthMapFolder", po::value<std::string>(&depthMapFolder)->required(),
            "Input depth maps folder.")
        ("depthMapFilterFolder", po::value<std::string>(&depthMapFilterFolder)->required(),
            "Input filtered depth maps folder.")
        ("output,o", po::value<std::string>(&outputMesh)->required(),
            "Output mesh (OBJ file format).");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("maxInputPoints", po::value<int>(&fuseParams.maxInputPoints)->default_value(fuseParams.maxInputPoints),
            "Max input points loaded from images.")
        ("maxPoints", po::value<int>(&fuseParams.maxPoints)->default_value(fuseParams.maxPoints),
            "Max points at the end of the depth maps fusion.")
        ("maxPointsPerVoxel", po::value<int>(&maxPtsPerVoxel)->default_value(maxPtsPerVoxel),
            "Max points per voxel.")
        ("minStep", po::value<int>(&fuseParams.minStep)->default_value(fuseParams.minStep),
            "The step used to load depth values from depth maps is computed from maxInputPts. Here we define the minimal value for this step, "
            "so on small datasets we will not spend too much time at the beginning loading all depth values.")
        ("simFactor", po::value<float>(&fuseParams.simFactor)->default_value(fuseParams.simFactor),
            "simFactor")
        ("angleFactor", po::value<float>(&fuseParams.angleFactor)->default_value(fuseParams.angleFactor),
            "angleFactor")
        ("partitioning", po::value<EPartitioningMode>(&partitioningMode)->default_value(partitioningMode),
            "Partitioning: 'singleBlock' or 'auto'.")
        ("repartition", po::value<ERepartitionMode>(&repartitionMode)->default_value(repartitionMode),
            "Repartition: 'multiResolution' or 'regularGrid'.");

    po::options_description advancedParams("Advanced parameters");
    advancedParams.add_options()
            ("pixSizeMarginInitCoef", po::value<double>(&fuseParams.pixSizeMarginInitCoef)->default_value(fuseParams.pixSizeMarginInitCoef),
                "pixSizeMarginInitCoef")
            ("pixSizeMarginFinalCoef", po::value<double>(&fuseParams.pixSizeMarginFinalCoef)->default_value(fuseParams.pixSizeMarginFinalCoef),
                "pixSizeMarginFinalCoef")
            ("voteMarginFactor", po::value<float>(&fuseParams.voteMarginFactor)->default_value(fuseParams.voteMarginFactor),
                "voteMarginFactor")
            ("contributeMarginFactor", po::value<float>(&fuseParams.contributeMarginFactor)->default_value(fuseParams.contributeMarginFactor),
                "contributeMarginFactor")
            ("simGaussianSizeInit", po::value<float>(&fuseParams.simGaussianSizeInit)->default_value(fuseParams.simGaussianSizeInit),
                "simGaussianSizeInit")
            ("simGaussianSize", po::value<float>(&fuseParams.simGaussianSize)->default_value(fuseParams.simGaussianSize),
                "simGaussianSize")
            ("minAngleThreshold", po::value<double>(&fuseParams.minAngleThreshold)->default_value(fuseParams.minAngleThreshold),
                "minAngleThreshold")
            ("refineFuse", po::value<bool>(&fuseParams.refineFuse)->default_value(fuseParams.refineFuse),
                "refineFuse");

    po::options_description logParams("Log parameters");
    logParams.add_options()
      ("verboseLevel,v", po::value<std::string>(&verboseLevel)->default_value(verboseLevel),
        "verbosity level (fatal, error, warning, info, debug, trace).");

    allParams.add(requiredParams).add(optionalParams).add(advancedParams).add(logParams);

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

    // .ini and files parsing
    mvsUtils::MultiViewParams mp(iniFilepath, depthMapFolder, depthMapFilterFolder, true);
    mvsUtils::PreMatchCams pc(&mp);

    int ocTreeDim = mp._ini.get<int>("LargeScale.gridLevel0", 1024);
    const auto baseDir = mp._ini.get<std::string>("LargeScale.baseDirName", "root01024");

    bfs::path outDirectory = bfs::path(outputMesh).parent_path();
    if(!bfs::is_directory(outDirectory))
        bfs::create_directory(outDirectory);

    bfs::path tmpDirectory = outDirectory / "tmp";

    ALICEVISION_LOG_WARNING("repartitionMode: " << repartitionMode);
    ALICEVISION_LOG_WARNING("partitioningMode: " << partitioningMode);

    switch(repartitionMode)
    {
        case eRepartitionRegularGrid:
        {
            switch(partitioningMode)
            {
                case ePartitioningAuto:
                {
                    ALICEVISION_LOG_INFO("Meshing mode: regular Grid, partitioning: auto.");
                    fuseCut::LargeScale lsbase(&mp, &pc, tmpDirectory.string() + "/");
                    lsbase.generateSpace(maxPtsPerVoxel, ocTreeDim, true);
                    std::string voxelsArrayFileName = lsbase.spaceFolderName + "hexahsToReconstruct.bin";
                    StaticVector<Point3d>* voxelsArray = nullptr;
                    if(bfs::exists(voxelsArrayFileName))
                    {
                        // If already computed reload it.
                        ALICEVISION_LOG_INFO("Voxels array already computed, reload from file: " << voxelsArrayFileName);
                        voxelsArray = loadArrayFromFile<Point3d>(voxelsArrayFileName);
                    }
                    else
                    {
                        ALICEVISION_LOG_INFO("Compute voxels array.");
                        fuseCut::ReconstructionPlan rp(lsbase.dimensions, &lsbase.space[0], lsbase.mp, lsbase.pc, lsbase.spaceVoxelsFolderName);
                        voxelsArray = rp.computeReconstructionPlanBinSearch(fuseParams.maxPoints);
                        saveArrayToFile<Point3d>(voxelsArrayFileName, voxelsArray);
                    }
                    fuseCut::reconstructSpaceAccordingToVoxelsArray(voxelsArrayFileName, &lsbase);
                    // Join meshes
                    mesh::Mesh* mesh = fuseCut::joinMeshes(voxelsArrayFileName, &lsbase);

                    if(mesh->pts->empty() || mesh->tris->empty())
                      throw std::runtime_error("Empty mesh");

                    ALICEVISION_LOG_INFO("Saving joined meshes...");

                    bfs::path spaceBinFileName = outDirectory/"denseReconstruction.bin";
                    bfs::path spaceXYZFileName = outDirectory / "denseReconstruction.xyz";
					//TODO@Yury get point cloud in correct format
                    mesh->saveToBin(spaceBinFileName.string());
                    mesh->saveToXYZ(spaceXYZFileName.string());
                    // Export joined mesh to obj
                    mesh->saveToObj(outputMesh);

                    delete mesh;

                    // Join ptsCams
                    StaticVector<StaticVector<int>*>* ptsCams = fuseCut::loadLargeScalePtsCams(lsbase.getRecsDirs(voxelsArray));
                    saveArrayOfArraysToFile<int>((outDirectory/"meshPtsCamsFromDGC.bin").string(), ptsCams);
                    deleteArrayOfArrays<int>(&ptsCams);
                    break;
                }
                case ePartitioningSingleBlock:
                {
                    ALICEVISION_LOG_INFO("Meshing mode: regular Grid, partitioning: single block.");
                    fuseCut::LargeScale ls0(&mp, &pc, tmpDirectory.string() + "/");
                    ls0.generateSpace(maxPtsPerVoxel, ocTreeDim, true);
                    unsigned long ntracks = std::numeric_limits<unsigned long>::max();
                    while(ntracks > fuseParams.maxPoints)
                    {
                        bfs::path dirName = outDirectory/("LargeScaleMaxPts" + mvsUtils::num2strFourDecimal(ocTreeDim));
                        fuseCut::LargeScale* ls = ls0.cloneSpaceIfDoesNotExists(ocTreeDim, dirName.string() + "/");
                        fuseCut::VoxelsGrid vg(ls->dimensions, &ls->space[0], ls->mp, ls->pc, ls->spaceVoxelsFolderName);
                        ntracks = vg.getNTracks();
                        delete ls;
                        ALICEVISION_LOG_INFO("Number of track candidates: " << ntracks);
                        if(ntracks > fuseParams.maxPoints)
                        {
                            ALICEVISION_LOG_INFO("ocTreeDim: " << ocTreeDim);
                            double t = (double)ntracks / (double)fuseParams.maxPoints;
                            ALICEVISION_LOG_INFO("downsample: " << ((t < 2.0) ? "slow" : "fast"));
                            ocTreeDim = (t < 2.0) ? ocTreeDim-100 : ocTreeDim*0.5;
                        }
                    }
                    ALICEVISION_LOG_INFO("Number of tracks: " << ntracks);
                    ALICEVISION_LOG_INFO("ocTreeDim: " << ocTreeDim);
                    bfs::path dirName = outDirectory/("LargeScaleMaxPts" + mvsUtils::num2strFourDecimal(ocTreeDim));
                    fuseCut::LargeScale lsbase(&mp, &pc, dirName.string()+"/");
                    lsbase.loadSpaceFromFile();
                    fuseCut::ReconstructionPlan rp(lsbase.dimensions, &lsbase.space[0], lsbase.mp, lsbase.pc, lsbase.spaceVoxelsFolderName);

                    StaticVector<int> voxelNeighs;
                    voxelNeighs.resize(rp.voxels->size() / 8);
                    ALICEVISION_LOG_INFO("voxelNeighs.size(): " << voxelNeighs.size());
                    for(int i = 0; i < voxelNeighs.size(); ++i)
                        voxelNeighs[i] = i;

                    fuseCut::DelaunayGraphCut delaunayGC(lsbase.mp, lsbase.pc);
                    Point3d* hexah = &lsbase.space[0];
                    delaunayGC.reconstructVoxel(hexah, &voxelNeighs, outDirectory.string()+"/", lsbase.getSpaceCamsTracksDir(), false,
                                          (fuseCut::VoxelsGrid*)&rp, lsbase.getSpaceSteps(), fuseParams);

                    delaunayGC.graphCutPostProcessing();

                    // Save mesh as .bin and .obj
                    mesh::Mesh* mesh = delaunayGC.createMesh();
                    if(mesh->pts->empty() || mesh->tris->empty())
                      throw std::runtime_error("Empty mesh");

                    StaticVector<StaticVector<int>*>* ptsCams = delaunayGC.createPtsCams();
                    StaticVector<int> usedCams = delaunayGC.getSortedUsedCams();

                    StaticVector<Point3d>* hexahsToExcludeFromResultingMesh = nullptr;
                    mesh::meshPostProcessing(mesh, ptsCams, usedCams, mp, pc, outDirectory.string()+"/", hexahsToExcludeFromResultingMesh, hexah);
                    mesh->saveToBin((outDirectory/"denseReconstruction.bin").string());
                    mesh->saveToXYZ((outDirectory/"denseReconstruction.xyz").string());

                    saveArrayOfArraysToFile<int>((outDirectory/"meshPtsCamsFromDGC.bin").string(), ptsCams);
                    deleteArrayOfArrays<int>(&ptsCams);

                    mesh->saveToObj(outputMesh);

                    delete mesh;
                    break;
                }
                case ePartitioningUndefined:
                default:
                    throw std::invalid_argument("Partitioning mode is not defined");
            }
            break;
        }
        case eRepartitionMultiResolution:
        {
            switch(partitioningMode)
            {
                case ePartitioningAuto:
                {
                    throw std::invalid_argument("Meshing mode: 'multiResolution', partitioning: 'auto' is not yet implemented.");
                }
                case ePartitioningSingleBlock:
                {
                    ALICEVISION_LOG_INFO("Meshing mode: multi-resolution, partitioning: single block.");
                    fuseCut::DelaunayGraphCut delaunayGC(&mp, &pc);
                    std::array<Point3d, 8> hexah;

                    float minPixSize;
                    fuseCut::Fuser fs(&mp, &pc);
                    fs.divideSpace(&hexah[0], minPixSize);
                    Voxel dimensions = fs.estimateDimensions(&hexah[0], &hexah[0], 0, ocTreeDim);
                    StaticVector<Point3d>* voxels = mvsUtils::computeVoxels(&hexah[0], dimensions);

                    StaticVector<int> voxelNeighs;
                    voxelNeighs.resize(voxels->size() / 8);
                    ALICEVISION_LOG_INFO("voxelNeighs.size(): " << voxelNeighs.size());
                    for(int i = 0; i < voxelNeighs.size(); ++i)
                        voxelNeighs[i] = i;
                    Point3d spaceSteps;
                    {
                        Point3d vx = hexah[1] - hexah[0];
                        Point3d vy = hexah[3] - hexah[0];
                        Point3d vz = hexah[4] - hexah[0];
                        spaceSteps.x = (vx.size() / (double)dimensions.x) / (double)ocTreeDim;
                        spaceSteps.y = (vy.size() / (double)dimensions.y) / (double)ocTreeDim;
                        spaceSteps.z = (vz.size() / (double)dimensions.z) / (double)ocTreeDim;
                    }
                    delaunayGC.reconstructVoxel(&hexah[0], &voxelNeighs, outDirectory.string()+"/", outDirectory.string()+"/SpaceCamsTracks/", false,
                                                nullptr, spaceSteps, fuseParams);
                    // TODO change the function name: reconstructFromDepthMaps(hexah);

                    delaunayGC.graphCutPostProcessing();

                    // Save mesh as .bin and .obj
                    mesh::Mesh* mesh = delaunayGC.createMesh();
                    if(mesh->pts->empty() || mesh->tris->empty())
                        throw std::runtime_error("Empty mesh");
					// NOTE@Yury this path is used by the mechroom
                    StaticVector<StaticVector<int>*>* ptsCams = delaunayGC.createPtsCams();
                    StaticVector<int> usedCams = delaunayGC.getSortedUsedCams();

                    StaticVector<Point3d>* hexahsToExcludeFromResultingMesh = nullptr;
                    mesh::meshPostProcessing(mesh, ptsCams, usedCams, mp, pc, outDirectory.string()+"/", hexahsToExcludeFromResultingMesh, &hexah[0]);
                    mesh->saveToBin((outDirectory/"denseReconstruction.bin").string());
                    mesh->saveToXYZ((outDirectory / "denseReconstruction.xyz").string());

                    saveArrayOfArraysToFile<int>((outDirectory/"meshPtsCamsFromDGC.bin").string(), ptsCams);
                    deleteArrayOfArrays<int>(&ptsCams);
                    delete voxels;

                    mesh->saveToObj(outputMesh);

                    delete mesh;
                    break;
                }
                case ePartitioningUndefined:
                default:
                    throw std::invalid_argument("Partitioning mode is not defined");
            }
            break;
        }
        case eRepartitionUndefined:
        default:
            throw std::invalid_argument("Repartition mode is not defined");
    }

    ALICEVISION_LOG_INFO("Task done in (s): " + std::to_string(timer.elapsed()));
    return EXIT_SUCCESS;
}
