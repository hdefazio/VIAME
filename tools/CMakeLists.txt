###
# This file contains an assortment of VIAME standalone tools
##

###
# Python standalone tools
##
set( PYTHON_SCRIPTS
  check_formatting.py
  database_tool.py
  extract_video_frames.py
  process_video.py
  launch_search_interface.py
  launch_timeline_interface.py
  launch_annotation_interface.py
  generate_detection_plots.py
  score_results.py )

if( VIAME_ENABLE_OPENCV )
  set( PYTHON_SCRIPTS
    ${PYTHON_SCRIPTS}
    calibrate_cameras.py
    create_mosaic.py
    compute_depth.py
    compute_disparity.py
    ply_utilities.py
    stereo_rectify.py )
endif()

if( VIAME_ENABLE_PYTORCH )
  set( PYTHON_SCRIPTS
    ${PYTHON_SCRIPTS}
    check_gpu_usability.py )
endif()

###
# C/C++ compiled standalone tools
##
kwiver_add_executable( viame_train_detector
  viame_train_detector.cxx
  )

target_link_libraries( viame_train_detector
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_algo
               kwiver::vital_util
               kwiver::kwiversys
               kwiver::sprokit_pipeline
               kwiver::kwiver_adapter
  )

if( VIAME_ENABLE_PYTHON )
  install( FILES       ${PYTHON_SCRIPTS}
           DESTINATION configs )
endif()
