## AliceVision
## Pipeline softwares

# Software PROPERTY FOLDER is 'Software/Pipeline'
set(FOLDER_SOFTWARE_PIPELINE "Software/ZMQ_workers")



### MVS software
if(ALICEVISION_BUILD_MVS)
  find_package(ZeroMQ CONFIG REQUIRED)

  
  # Depth Map Estimation
  if(ALICEVISION_HAVE_CUDA) # Depth map computation need CUDA
    alicevision_add_software(aliceVision_ventilator_depthMapEstimation
      SOURCE ventilator_depthMapEstimation.cpp
      FOLDER ${FOLDER_SOFTWARE_PIPELINE}
      LINKS aliceVision_system
          aliceVision_mvsData
          aliceVision_mvsUtils
          aliceVision_depthMap
          ${Boost_LIBRARIES}
          libzmq 
          libzmq-static
      INCLUDE_DIRS ${ZeroMQ_INCLUDE_DIR}
      )
    alicevision_add_software(aliceVision_worker_depthMapEstimation
      SOURCE worker_depthMapEstimation.cpp
      FOLDER ${FOLDER_SOFTWARE_PIPELINE}
      LINKS aliceVision_system
            aliceVision_mvsData
            aliceVision_mvsUtils
            aliceVision_depthMap
            ${Boost_LIBRARIES}
            libzmq 
            libzmq-static
      INCLUDE_DIRS ${ZeroMQ_INCLUDE_DIR}
      )
  endif()

   # Depth Map Filtering
  alicevision_add_software(aliceVision_ventilator_depthMapFiltering
    SOURCE ventilator_depthMapFiltering.cpp
    FOLDER ${FOLDER_SOFTWARE_PIPELINE}
    LINKS aliceVision_system
          aliceVision_mvsData
          aliceVision_mvsUtils
          aliceVision_fuseCut
          ${Boost_LIBRARIES}
          libzmq 
          libzmq-static
    INCLUDE_DIRS ${ZeroMQ_INCLUDE_DIR}
 )

  # Depth Map Filtering
  alicevision_add_software(aliceVision_worker_depthMapFiltering
    SOURCE worker_depthMapFiltering.cpp
    FOLDER ${FOLDER_SOFTWARE_PIPELINE}
    LINKS aliceVision_system
          aliceVision_mvsData
          aliceVision_mvsUtils
          aliceVision_fuseCut
          ${Boost_LIBRARIES}
          libzmq 
          libzmq-static
    INCLUDE_DIRS ${ZeroMQ_INCLUDE_DIR}
  )


endif() # if(ALICEVISION_BUILD_MVS)
