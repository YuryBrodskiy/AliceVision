// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "colorize.hpp"
#include <aliceVision/alicevision_omp.hpp>
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/stl/indexedSort.hpp>
#include <aliceVision/stl/mapUtils.hpp>
#include <aliceVision/image/io.hpp>

#include <boost/progress.hpp>

#include <map>
#include <vector>
#include <functional>
namespace aliceVision {
namespace sfmData {

void colorizeTracks(SfMData& sfmData)
{
  boost::progress_display progressBar(sfmData.getLandmarks().size(), std::cout, "\nCompute scene structure color\n");

  std::vector<std::reference_wrapper<Landmark>> remainingLandmarksToColor;
  remainingLandmarksToColor.reserve(sfmData.getLandmarks().size());

  for(auto& landmarkPair : sfmData.getLandmarks())
    remainingLandmarksToColor.push_back(landmarkPair.second);

  struct ViewInfo
  {
    ViewInfo(IndexT viewId, std::size_t cardinal)
      : viewId(viewId)
      , cardinal(cardinal)
    {}

    IndexT viewId;
    std::size_t cardinal;
    std::vector<std::reference_wrapper<Landmark>> landmarks;
  };

  std::vector<ViewInfo> sortedViewsCardinal;
  sortedViewsCardinal.reserve(sfmData.getViews().size());
  {
    // create cardinal per viewId map
    std::map<IndexT, std::size_t> viewsCardinalMap; // <ViewId, Cardinal>
    for(const auto& landmarkPair : sfmData.getLandmarks())
    {
      const Observations& observations = landmarkPair.second.observations;
      for(const auto& observationPair : observations)
        ++viewsCardinalMap[observationPair.first]; // TODO: 0
    }

    // copy key-value pairs from the map to the vector
    for(const auto& cardinalPair : viewsCardinalMap)
      sortedViewsCardinal.push_back(ViewInfo(cardinalPair.first, cardinalPair.second));

    // sort the vector, biggest cardinality first
    std::sort(sortedViewsCardinal.begin(),
              sortedViewsCardinal.end(),
              [] (const ViewInfo& l, const ViewInfo& r) { return l.cardinal > r.cardinal; });
  }

  // assign each landmark to a view
  for(ViewInfo& viewCardinal : sortedViewsCardinal)
  {
    std::vector<std::reference_wrapper<Landmark>> toKeep;
    const IndexT viewId = viewCardinal.viewId;

    for(int i = 0; i < remainingLandmarksToColor.size(); ++i)
    {
      Landmark& landmark = remainingLandmarksToColor.at(i);
      auto it = landmark.observations.find(viewId);
      if(it != landmark.observations.end())
      {
        viewCardinal.landmarks.push_back(landmark);
      }
      else
      {
        toKeep.push_back(landmark);
      }
    }
    std::swap(toKeep, remainingLandmarksToColor);

    if(remainingLandmarksToColor.empty())
      break;
  }

  // create an unsorted index container
  std::vector<int> unsortedIndexes(sortedViewsCardinal.size()) ;
  std::iota(std::begin(unsortedIndexes), std::end(unsortedIndexes), 0);
  std::random_shuffle(unsortedIndexes.begin(), unsortedIndexes.end());

  // landmark colorization
#pragma omp parallel for
  for(int i = 0; i < unsortedIndexes.size(); ++i)
  {
    const ViewInfo& viewCardinal = sortedViewsCardinal.at(unsortedIndexes.at(i));
    if(!viewCardinal.landmarks.empty())
    {
      const View& view = sfmData.getView(viewCardinal.viewId);
      image::Image<image::RGBColor> image;
      image::readImage(view.getImagePath(), image, image::EImageColorSpace::SRGB);

      for(Landmark& landmark : viewCardinal.landmarks)
      {
        // color the point
        Vec2 pt = landmark.observations.at(view.getViewId()).x;
        // clamp the pixel position if the feature/marker center is outside the image.
        pt.x() = clamp(pt.x(), 0.0, static_cast<double>(image.Width() - 1));
        pt.y() = clamp(pt.y(), 0.0, static_cast<double>(image.Height() - 1));
        landmark.rgb = image(pt.y(), pt.x());
      }

#pragma omp critical
      {
        progressBar += viewCardinal.landmarks.size();
      }
    }
  }
}

void colorizeTracksV2(SfMData& sfmData) 
{
    boost::progress_display progressBar(sfmData.getViews().size(), std::cout, "\nCompute scene structure color\n");
    std::map <IndexT, std::vector<Landmark*>> lm_map;
    for(auto& lm : sfmData.getLandmarks())
    {
        lm.second.rgb = image::BLACK;
        for(const auto& proj : lm.second.observations)
        {
            const auto& viewId = proj.first;
            lm_map[viewId].push_back(&lm.second);
        }
    }

    for(auto& pair : lm_map)
    {
        progressBar += 1; 
        const auto& viewId = pair.first;
        const View& view = sfmData.getView(viewId);
        image::Image<image::RGBColor> image;
        image::readImage(view.getImagePath(), image, image::EImageColorSpace::SRGB);
        // for(Landmark* lm : pair.second)
#pragma omp parallel for
        for(int i = 0; i < pair.second.size(); ++i)
        {
            Landmark* lm = pair.second[i];
            auto proj = lm->observations.at(viewId);
            Vec2 pt = proj.x;
            if(pt.x() == clamp(pt.x(), 0.0, static_cast<double>(image.Width() - 1)) &&
               pt.y() == clamp(pt.y(), 0.0, static_cast<double>(image.Height() - 1)) )
            {
                lm->rgb += image(pt.y(), pt.x()) / lm->observations.size();
            }
        }
    }

}

} // namespace sfm
} // namespace aliceVision
