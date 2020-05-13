// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "jsonIO.hpp"
#include <aliceVision/camera/camera.hpp>
#include <aliceVision/sfmDataIO/viewIO.hpp>

#include <boost/property_tree/json_parser.hpp>

#include <memory>
#include <cassert>
#include <iosfwd>

namespace aliceVision
{
namespace sfmDataIO
{

void saveView(const std::string& name, const sfmData::View& view, bpt::ptree& parentTree)
{
    bpt::ptree viewTree;

    if(view.getViewId() != UndefinedIndexT)
        viewTree.put("viewId", view.getViewId());

    if(view.getPoseId() != UndefinedIndexT)
        viewTree.put("poseId", view.getPoseId());

    if(view.isPartOfRig())
    {
        viewTree.put("rigId", view.getRigId());
        viewTree.put("subPoseId", view.getSubPoseId());
    }

    if(view.getFrameId() != UndefinedIndexT)
        viewTree.put("frameId", view.getFrameId());

    if(view.getIntrinsicId() != UndefinedIndexT)
        viewTree.put("intrinsicId", view.getIntrinsicId());

    if(view.getResectionId() != UndefinedIndexT)
        viewTree.put("resectionId", view.getResectionId());

    if(view.isPoseIndependant() == false)
        viewTree.put("isPoseIndependant", view.isPoseIndependant());

    viewTree.put("path", view.getImagePath());
    viewTree.put("width", view.getWidth());
    viewTree.put("height", view.getHeight());

    // metadata
    {
        bpt::ptree metadataTree;

        for(const auto& metadataPair : view.getMetadata())
            metadataTree.put(metadataPair.first, metadataPair.second);

        viewTree.add_child("metadata", metadataTree);
    }

    parentTree.push_back(std::make_pair(name, viewTree));
}

void loadView(sfmData::View& view, bpt::ptree& viewTree)
{
    view.setViewId(viewTree.get<IndexT>("viewId", UndefinedIndexT));
    view.setPoseId(viewTree.get<IndexT>("poseId", UndefinedIndexT));

    if(viewTree.count("rigId"))
    {
        view.setRigAndSubPoseId(viewTree.get<IndexT>("rigId"), viewTree.get<IndexT>("subPoseId"));
    }

    view.setFrameId(viewTree.get<IndexT>("frameId", UndefinedIndexT));
    view.setIntrinsicId(viewTree.get<IndexT>("intrinsicId", UndefinedIndexT));
    view.setResectionId(viewTree.get<IndexT>("resectionId", UndefinedIndexT));
    view.setIndependantPose(viewTree.get<bool>("isPoseIndependant", true));

    view.setImagePath(viewTree.get<std::string>("path"));
    view.setWidth(viewTree.get<std::size_t>("width", 0));
    view.setHeight(viewTree.get<std::size_t>("height", 0));

    // metadata
    if(viewTree.count("metadata"))
        for(bpt::ptree::value_type& metaDataNode : viewTree.get_child("metadata"))
            view.addMetadata(metaDataNode.first, metaDataNode.second.data());
}

void saveIntrinsic(const std::string& name, IndexT intrinsicId, const std::shared_ptr<camera::IntrinsicBase>& intrinsic,
                   bpt::ptree& parentTree)
{
    bpt::ptree intrinsicTree;

    camera::EINTRINSIC intrinsicType = intrinsic->getType();

    intrinsicTree.put("intrinsicId", intrinsicId);
    intrinsicTree.put("width", intrinsic->w());
    intrinsicTree.put("height", intrinsic->h());
    intrinsicTree.put("serialNumber", intrinsic->serialNumber());
    intrinsicTree.put("type", camera::EINTRINSIC_enumToString(intrinsicType));
    intrinsicTree.put("initializationMode",
                      camera::EIntrinsicInitMode_enumToString(intrinsic->getInitializationMode()));
    intrinsicTree.put("pxInitialFocalLength", intrinsic->initialFocalLengthPix());

    if(camera::isPinhole(intrinsicType))
    {
        const camera::Pinhole& pinholeIntrinsic = dynamic_cast<camera::Pinhole&>(*intrinsic);

        intrinsicTree.put("pxFocalLength", pinholeIntrinsic.getFocalLengthPix());
        saveMatrix("principalPoint", pinholeIntrinsic.getPrincipalPoint(), intrinsicTree);

        bpt::ptree distParamsTree;

        for(double param : pinholeIntrinsic.getDistortionParams())
        {
            bpt::ptree paramTree;
            paramTree.put("", param);
            distParamsTree.push_back(std::make_pair("", paramTree));
        }

        intrinsicTree.add_child("distortionParams", distParamsTree);
    }

    intrinsicTree.put("locked",
                      static_cast<int>(intrinsic->isLocked())); // convert bool to integer to avoid using "true/false"
                                                                // in exported file instead of "1/0".

    parentTree.push_back(std::make_pair(name, intrinsicTree));
}

void loadIntrinsic(IndexT& intrinsicId, std::shared_ptr<camera::IntrinsicBase>& intrinsic, bpt::ptree& intrinsicTree)
{
    intrinsicId = intrinsicTree.get<IndexT>("intrinsicId");
    const unsigned int width = intrinsicTree.get<unsigned int>("width");
    const unsigned int height = intrinsicTree.get<unsigned int>("height");
    const camera::EINTRINSIC intrinsicType = camera::EINTRINSIC_stringToEnum(intrinsicTree.get<std::string>("type"));
    const camera::EIntrinsicInitMode initializationMode =
        camera::EIntrinsicInitMode_stringToEnum(intrinsicTree.get<std::string>(
            "initializationMode", camera::EIntrinsicInitMode_enumToString(camera::EIntrinsicInitMode::CALIBRATED)));
    const double pxFocalLength = intrinsicTree.get<double>("pxFocalLength");

    // principal point
    Vec2 principalPoint;
    loadMatrix("principalPoint", principalPoint, intrinsicTree);

    // check if the camera is a Pinhole model
    if(!camera::isPinhole(intrinsicType))
        throw std::out_of_range("Only Pinhole camera model supported");

    // pinhole parameters
    std::shared_ptr<camera::Pinhole> pinholeIntrinsic = camera::createPinholeIntrinsic(
        intrinsicType, width, height, pxFocalLength, principalPoint(0), principalPoint(1));
    pinholeIntrinsic->setInitialFocalLengthPix(intrinsicTree.get<double>("pxInitialFocalLength"));
    pinholeIntrinsic->setSerialNumber(intrinsicTree.get<std::string>("serialNumber"));
    pinholeIntrinsic->setInitializationMode(initializationMode);

    std::vector<double> distortionParams;
    for(bpt::ptree::value_type& paramNode : intrinsicTree.get_child("distortionParams"))
        distortionParams.emplace_back(paramNode.second.get_value<double>());

    // ensure that we have the right number of params
    distortionParams.resize(pinholeIntrinsic->getDistortionParams().size(), 0.0);

    pinholeIntrinsic->setDistortionParams(distortionParams);
    intrinsic = std::static_pointer_cast<camera::IntrinsicBase>(pinholeIntrinsic);

    // intrinsic lock
    if(intrinsicTree.get<bool>("locked", false))
        intrinsic->lock();
    else
        intrinsic->unlock();
}

void saveRig(const std::string& name, IndexT rigId, const sfmData::Rig& rig, bpt::ptree& parentTree)
{
    bpt::ptree rigTree;

    rigTree.put("rigId", rigId);

    bpt::ptree rigSubPosesTree;

    for(const auto& rigSubPose : rig.getSubPoses())
    {
        bpt::ptree rigSubPoseTree;

        rigSubPoseTree.put("status", sfmData::ERigSubPoseStatus_enumToString(rigSubPose.status));
        savePose3("pose", rigSubPose.pose, rigSubPoseTree);

        rigSubPosesTree.push_back(std::make_pair("", rigSubPoseTree));
    }

    rigTree.add_child("subPoses", rigSubPosesTree);

    parentTree.push_back(std::make_pair(name, rigTree));
}

void loadRig(IndexT& rigId, sfmData::Rig& rig, bpt::ptree& rigTree)
{
    rigId = rigTree.get<IndexT>("rigId");
    rig = sfmData::Rig(rigTree.get_child("subPoses").size());
    int subPoseId = 0;

    for(bpt::ptree::value_type& subPoseNode : rigTree.get_child("subPoses"))
    {
        bpt::ptree& subPoseTree = subPoseNode.second;

        sfmData::RigSubPose subPose;

        subPose.status = sfmData::ERigSubPoseStatus_stringToEnum(subPoseTree.get<std::string>("status"));
        loadPose3("pose", subPose.pose, subPoseTree);

        rig.setSubPose(subPoseId++, subPose);
    }
}

void saveLandmark(const std::string& name, IndexT landmarkId, const sfmData::Landmark& landmark, bpt::ptree& parentTree,
                  bool saveObservations, bool saveFeatures)
{
    bpt::ptree landmarkTree;

    landmarkTree.put("landmarkId", landmarkId);
    landmarkTree.put("descType", feature::EImageDescriberType_enumToString(landmark.descType));

    saveMatrix("color", landmark.rgb, landmarkTree);
    saveMatrix("X", landmark.X, landmarkTree);

    // observations
    if(saveObservations)
    {
        bpt::ptree observationsTree;
        for(const auto& obsPair : landmark.observations)
        {
            bpt::ptree obsTree;

            const sfmData::Observation& observation = obsPair.second;

            obsTree.put("observationId", obsPair.first);

            // features
            if(saveFeatures)
            {
                obsTree.put("featureId", observation.id_feat);
                saveMatrix("x", observation.x, obsTree);
            }

            observationsTree.push_back(std::make_pair("", obsTree));
        }

        landmarkTree.add_child("observations", observationsTree);
    }

    parentTree.push_back(std::make_pair(name, landmarkTree));
}

void loadLandmark(IndexT& landmarkId, sfmData::Landmark& landmark, bpt::ptree& landmarkTree, bool loadObservations,
                  bool loadFeatures)
{
    landmarkId = landmarkTree.get<IndexT>("landmarkId");
    landmark.descType = feature::EImageDescriberType_stringToEnum(landmarkTree.get<std::string>("descType"));

    loadMatrix("color", landmark.rgb, landmarkTree);
    loadMatrix("X", landmark.X, landmarkTree);

    // observations
    if(loadObservations)
    {
        for(bpt::ptree::value_type& obsNode : landmarkTree.get_child("observations"))
        {
            bpt::ptree& obsTree = obsNode.second;

            sfmData::Observation observation;

            if(loadFeatures)
            {
                observation.id_feat = obsTree.get<IndexT>("featureId");
                loadMatrix("x", observation.x, obsTree);
            }

            landmark.observations.emplace(obsTree.get<IndexT>("observationId"), observation);
        }
    }
}

bool saveJSON(const sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag)
{
    const Vec3 version = {1, 0, 0};

    // save flags
    const bool saveViews = (partFlag & VIEWS) == VIEWS;
    const bool saveIntrinsics = (partFlag & INTRINSICS) == INTRINSICS;
    const bool saveExtrinsics = (partFlag & EXTRINSICS) == EXTRINSICS;
    const bool saveStructure = (partFlag & STRUCTURE) == STRUCTURE;
    const bool saveControlPoints = (partFlag & CONTROL_POINTS) == CONTROL_POINTS;
    const bool saveFeatures = (partFlag & OBSERVATIONS_WITH_FEATURES) == OBSERVATIONS_WITH_FEATURES;
    const bool saveObservations = saveFeatures || ((partFlag & OBSERVATIONS) == OBSERVATIONS);

    // main tree
    bpt::ptree fileTree;

    // file version
    saveMatrix("version", version, fileTree);

    // folders
    if(!sfmData.getRelativeFeaturesFolders().empty())
    {
        bpt::ptree featureFoldersTree;

        for(const std::string& featuresFolder : sfmData.getRelativeFeaturesFolders())
        {
            bpt::ptree featureFolderTree;
            featureFolderTree.put("", featuresFolder);
            featureFoldersTree.push_back(std::make_pair("", featureFolderTree));
        }

        fileTree.add_child("featuresFolders", featureFoldersTree);
    }

    if(!sfmData.getRelativeMatchesFolders().empty())
    {
        bpt::ptree matchingFoldersTree;

        for(const std::string& matchesFolder : sfmData.getRelativeMatchesFolders())
        {
            bpt::ptree matchingFolderTree;
            matchingFolderTree.put("", matchesFolder);
            matchingFoldersTree.push_back(std::make_pair("", matchingFolderTree));
        }

        fileTree.add_child("matchesFolders", matchingFoldersTree);
    }

    // views
    if(saveViews && !sfmData.getViews().empty())
    {
        bpt::ptree viewsTree;

        for(const auto& viewPair : sfmData.getViews())
            saveView("", *(viewPair.second), viewsTree);

        fileTree.add_child("views", viewsTree);
    }

    // intrinsics
    if(saveIntrinsics && !sfmData.getIntrinsics().empty())
    {
        bpt::ptree intrinsicsTree;

        for(const auto& intrinsicPair : sfmData.getIntrinsics())
            saveIntrinsic("", intrinsicPair.first, intrinsicPair.second, intrinsicsTree);

        fileTree.add_child("intrinsics", intrinsicsTree);
    }

    // extrinsics
    if(saveExtrinsics)
    {
        // poses
        if(!sfmData.getPoses().empty())
        {
            bpt::ptree posesTree;

            for(const auto& posePair : sfmData.getPoses())
            {
                bpt::ptree poseTree;

                poseTree.put("poseId", posePair.first);
                saveCameraPose("pose", posePair.second, poseTree);
                posesTree.push_back(std::make_pair("", poseTree));
            }

            fileTree.add_child("poses", posesTree);
        }

        // rigs
        if(!sfmData.getRigs().empty())
        {
            bpt::ptree rigsTree;

            for(const auto& rigPair : sfmData.getRigs())
                saveRig("", rigPair.first, rigPair.second, rigsTree);

            fileTree.add_child("rigs", rigsTree);
        }
    }

    // structure
    if(saveStructure && !sfmData.getLandmarks().empty())
    {
        bpt::ptree structureTree;

        for(const auto& structurePair : sfmData.getLandmarks())
            saveLandmark("", structurePair.first, structurePair.second, structureTree, saveObservations, saveFeatures);

        fileTree.add_child("structure", structureTree);
    }

    // control points
    if(saveControlPoints && !sfmData.getControlPoints().empty())
    {
        bpt::ptree controlPointTree;

        for(const auto& controlPointPair : sfmData.getControlPoints())
            saveLandmark("", controlPointPair.first, controlPointPair.second, controlPointTree);

        fileTree.add_child("controlPoints", controlPointTree);
    }

    // write the json file with the tree

    bpt::write_json(filename, fileTree);

    return true;
}

bool loadJSON(sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag, bool incompleteViews)
{
    Vec3 version;

    // load flags
    const bool loadViews = (partFlag & VIEWS) == VIEWS;
    const bool loadIntrinsics = (partFlag & INTRINSICS) == INTRINSICS;
    const bool loadExtrinsics = (partFlag & EXTRINSICS) == EXTRINSICS;
    const bool loadStructure = (partFlag & STRUCTURE) == STRUCTURE;
    const bool loadControlPoints = (partFlag & CONTROL_POINTS) == CONTROL_POINTS;
    const bool loadFeatures = (partFlag & OBSERVATIONS_WITH_FEATURES) == OBSERVATIONS_WITH_FEATURES;
    const bool loadObservations = loadFeatures || ((partFlag & OBSERVATIONS) == OBSERVATIONS);

    // main tree
    bpt::ptree fileTree;

    // read the json file and initialize the tree
    bpt::read_json(filename, fileTree);

    // version
    loadMatrix("version", version, fileTree);

    // folders
    if(fileTree.count("featuresFolders"))
        for(bpt::ptree::value_type& featureFolderNode : fileTree.get_child("featuresFolders"))
            sfmData.addFeaturesFolder(featureFolderNode.second.get_value<std::string>());

    if(fileTree.count("matchesFolders"))
        for(bpt::ptree::value_type& matchingFolderNode : fileTree.get_child("matchesFolders"))
            sfmData.addMatchesFolder(matchingFolderNode.second.get_value<std::string>());

    //ALICEVISION_LOG_DEBUG("Load intrisics");

    // intrinsics
    if(loadIntrinsics && fileTree.count("intrinsics"))
    {
        sfmData::Intrinsics& intrinsics = sfmData.getIntrinsics();

        for(bpt::ptree::value_type& intrinsicNode : fileTree.get_child("intrinsics"))
        {
            IndexT intrinsicId;
            std::shared_ptr<camera::IntrinsicBase> intrinsic;

            loadIntrinsic(intrinsicId, intrinsic, intrinsicNode.second);

            intrinsics.emplace(intrinsicId, intrinsic);
        }
    }

	//ALICEVISION_LOG_DEBUG("Load views");

    // views
    if(loadViews && fileTree.count("views"))
    {
        sfmData::Views& views = sfmData.getViews();

        if(incompleteViews)
        {
            // store incomplete views in a vector
            std::vector<sfmData::View> incompleteViews(fileTree.get_child("views").size());

            int viewIndex = 0;
            for(bpt::ptree::value_type& viewNode : fileTree.get_child("views"))
            {
                // TODO@Yury add path correction for relative paths viewTree.get<std::string>("path")

                loadView(incompleteViews.at(viewIndex), viewNode.second);
                // incompleteViews.at(viewIndex).setImagePath
                ++viewIndex;
            }

// update incomplete views
#pragma omp parallel for
            for(int i = 0; i < incompleteViews.size(); ++i)
            {
                sfmData::View& v = incompleteViews.at(i);
                // if we have the intrinsics and the view has an valid associated intrinsics
                // update the width and height field of View (they are mirrored)
                if(loadIntrinsics && v.getIntrinsicId() != UndefinedIndexT)
                {
                    const auto intrinsics = sfmData.getIntrinsicPtr(v.getIntrinsicId());

                    if(intrinsics == nullptr)
                    {
                        throw std::logic_error("View " + std::to_string(v.getViewId()) + " has a intrinsics id " +
                                               std::to_string(v.getIntrinsicId()) +
                                               " that cannot be found or the intrinsics are not correctly "
                                               "loaded from the json file.");
                    }

                    v.setWidth(intrinsics->w());
                    v.setHeight(intrinsics->h());
                }
                updateIncompleteView(incompleteViews.at(i));
            }

            // copy complete views in the SfMData views map
            for(const sfmData::View& view : incompleteViews)
                views.emplace(view.getViewId(), std::make_shared<sfmData::View>(view));
        }
        else
        {
            // store directly in the SfMData views map
            for(bpt::ptree::value_type& viewNode : fileTree.get_child("views"))
            {
                sfmData::View view;
                loadView(view, viewNode.second);
                views.emplace(view.getViewId(), std::make_shared<sfmData::View>(view));
            }
        }
    }

	//ALICEVISION_LOG_DEBUG("Load extrisics");

    // extrinsics
    if(loadExtrinsics)
    {
        // poses
        if(fileTree.count("poses"))
        {
            sfmData::Poses& poses = sfmData.getPoses();

            for(bpt::ptree::value_type& poseNode : fileTree.get_child("poses"))
            {
                bpt::ptree& poseTree = poseNode.second;
                sfmData::CameraPose pose;

                loadCameraPose("pose", pose, poseTree);

                poses.emplace(poseTree.get<IndexT>("poseId"), pose);
            }
        }

        // rigs
        if(fileTree.count("rigs"))
        {
            sfmData::Rigs& rigs = sfmData.getRigs();

            for(bpt::ptree::value_type& rigNode : fileTree.get_child("rigs"))
            {
                IndexT rigId;
                sfmData::Rig rig;

                loadRig(rigId, rig, rigNode.second);

                rigs.emplace(rigId, rig);
            }
        }
    }

	//ALICEVISION_LOG_DEBUG("Load structure");

    // structure
    if(loadStructure && fileTree.count("structure"))
    {
        sfmData::Landmarks& structure = sfmData.getLandmarks();

        for(bpt::ptree::value_type& landmarkNode : fileTree.get_child("structure"))
        {
            IndexT landmarkId;
            sfmData::Landmark landmark;

            loadLandmark(landmarkId, landmark, landmarkNode.second, loadObservations, loadFeatures);

            structure.emplace(landmarkId, landmark);
        }
    }

	//ALICEVISION_LOG_DEBUG("Load control points");

    // control points
    if(loadControlPoints && fileTree.count("controlPoints"))
    {
        sfmData::Landmarks& controlPoints = sfmData.getControlPoints();

        for(bpt::ptree::value_type& landmarkNode : fileTree.get_child("controlPoints"))
        {
            IndexT landmarkId;
            sfmData::Landmark landmark;

            loadLandmark(landmarkId, landmark, landmarkNode.second);

            controlPoints.emplace(landmarkId, landmark);
        }
    }

    return true;
}

bool saveXYZ(const sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag)
{
    std::ofstream f(filename);
    
    if(!sfmData.getLandmarks().empty())
    {
        for (const auto& structurePair : sfmData.getLandmarks())
        {
            auto landmark = structurePair.second;
            //std::cout << landmark.X.transpose() << " " << landmark.rgb.transpose().cast<int>() << std::endl;
            f << landmark.X.transpose() << " " << landmark.rgb.transpose().cast<int>() << std::endl;
                
        }
    }
    f.close();
    return true;
}

Eigen::Matrix4d getTransformToGlobalJson(const std::string& filename)
{
    Eigen::Matrix4d H_0_n0 = Eigen::Matrix4d::Identity();
    bpt::ptree fileTree;
    bpt::read_json(filename, fileTree);
    loadMatrix("H_0_n0", H_0_n0, fileTree);
    return H_0_n0.transpose(); // alice reads matrix column wise
}

Eigen::Matrix4d getTransformToGlobalText(const std::string& filename) 
{
    using namespace std;
    Eigen::Matrix4d H_0_n0 = Eigen::Matrix4d::Identity();
    ifstream inFile(filename);
    string line;
    short order = 0;

    while(getline(inFile, line))
    {
        if(line.empty())
            continue;

        istringstream iss(line);
        vector<string> pieces(istream_iterator<string>{iss}, istream_iterator<string>());

        if(pieces.size() == 4) // It has only x,y,z information saved
        {
            H_0_n0(order, 0) = stod(pieces[0].c_str());
            H_0_n0(order, 1) = stod(pieces[1].c_str());
            H_0_n0(order, 2) = stod(pieces[2].c_str());
            H_0_n0(order, 3) = stod(pieces[3].c_str());
            order++;
        }
    }
    return H_0_n0;
}

} // namespace sfmDataIO
} // namespace aliceVision
