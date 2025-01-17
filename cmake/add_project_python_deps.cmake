# ------------------------------------------------------------------------------------------------
# Python Dependency External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

# ------------------------------ ADD ANY BASIC PYTHON DEPS HERE ----------------------------------
# Basic dependencies are installed jointly in one local pip installation call

set( PYTHON_DEP_ENV_VARS )

if( VIAME_FIXUP_BUNDLE )
  set( VIAME_PYTHON_BASIC_DEPS "numpy==1.19.3" )
else()
  set( VIAME_PYTHON_BASIC_DEPS "numpy<=1.23.5" )
endif()

# Setuptools < 58.0 required for current version of gdal on windows
if( WIN32 AND VIAME_ENABLE_PYTORCH-NETHARN AND VIAME_ENABLE_GDAL )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "setuptools==57.5.0" )
endif()

# Installation requirement for some dependencies
list( APPEND VIAME_PYTHON_BASIC_DEPS "Cython" "ordered_set" )

# For scoring and plotting
list( APPEND VIAME_PYTHON_BASIC_DEPS "kiwisolver==1.2.0" )
list( APPEND VIAME_PYTHON_BASIC_DEPS "matplotlib<=3.5.1" )

# For netharn and mmdet de-pickle on older versions
if( Python_VERSION VERSION_LESS "3.8" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pickle5" )

  if( VIAME_ENABLE_PYTORCH-PYSOT )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "protobuf==3.19.4" )
  endif()
endif()

# For fusion classifier
list( APPEND VIAME_PYTHON_BASIC_DEPS "map_boxes" "ensemble_boxes" )

if( Python_VERSION VERSION_LESS "3.9" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS  "llvmlite==0.31.0" "numba==0.47" )
else()
  list( APPEND VIAME_PYTHON_BASIC_DEPS  "llvmlite==0.39.1" "numba==0.56" )
endif()

# For pytorch building
if( VIAME_ENABLE_PYTORCH-INTERNAL )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "typing-extensions" "bs4" )
endif()

# For mmdetection
if( VIAME_ENABLE_PYTORCH-MMDET )
  if( Python_VERSION VERSION_LESS "3.8" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "yapf<=0.32.0" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "yapf" )
  endif()
endif()

# For measurement scripts
if( VIAME_ENABLE_OPENCV )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "tqdm" "scipy" )
endif()

if( ( WIN32 OR NOT VIAME_ENABLE_OPENCV ) AND
      ( VIAME_ENABLE_OPENCV OR
        VIAME_ENABLE_PYTORCH-MMDET OR
        VIAME_ENABLE_PYTORCH-NETHARN ) )
  if( Python_VERSION VERSION_LESS "3.7" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "opencv-python<=4.6.0.66" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "opencv-python" )
  endif()
endif()

if( VIAME_ENABLE_ITK_EXTRAS )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "msgpack" )
endif()

if( VIAME_ENABLE_PYTORCH )
  if( Python_VERSION VERSION_LESS "3.10" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-image==0.16.2" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-image==0.19.2" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-MMDET OR VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-build" "async_generator" )
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN AND VIAME_ENABLE_GDAL )
  if( WIN32 )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "gdal==2.2.3" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "gdal" )
  endif()
endif()

if( VIAME_ENABLE_OPENCV OR VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "ubelt" "pygments" "bezier==2020.1.14" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "ndsampler==0.6.7" "kwcoco==0.2.31" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "imageio==2.15.0" "networkx<=2.8.8" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-INTERNAL )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pyyaml" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-MMDET )
  if( WIN32 )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "pycocotools-windows" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "pycocotools" )
  endif()
endif()

if( VIAME_ENABLE_PYTHON-INTERNAL AND UNIX )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "backports.lzma" "backports.weakref" )
endif()

if( VIAME_ENABLE_TENSORFLOW )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "humanfriendly" )
  set( TF_ARGS "" )

  if( VIAME_TENSORFLOW_VERSION VERSION_LESS "2.0" )
    if( VIAME_ENABLE_CUDA )
      set( TF_ARGS "-gpu" )
    endif()
  else()
    if( NOT VIAME_ENABLE_CUDA )
      set( TF_ARGS "-cpu" )
    endif()
  endif()

  set( TF_ARGS "${TF_ARGS}==${VIAME_TENSORFLOW_VERSION}" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "tensorflow${TF_ARGS}" )
endif()

# ------------------------------ ADD ANY ADV PYTHON DEPS HERE ------------------------------------
# Advanced python dependencies are installed individually due to special reqs

set( VIAME_PYTHON_ADV_DEPS python-deps )
set( VIAME_PYTHON_ADV_DEP_CMDS "custom-install" )

if( VIAME_ENABLE_ITK_EXTRAS )
  set( WX_VERSION "4.0.7" )

  list( APPEND VIAME_PYTHON_ADV_DEPS wxPython )

  if( UNIX )
    if( EXISTS "/etc/os-release" )
      ParseLinuxOSField( "ID" OS_ID )
    endif()

    execute_process( COMMAND lsb_release -cs
      OUTPUT_VARIABLE RELEASE_CODENAME
      OUTPUT_STRIP_TRAILING_WHITESPACE )

    if( "${OS_ID}" MATCHES "centos" )
      set( WXP_ARCHIVE https://extras.wxpython.org/wxPython4/extras/linux/gtk3/centos-7 )
      list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "-U -f ${WXP_ARCHIVE} wxPython==${WX_VERSION}" )
    elseif( "${RELEASE_CODENAME}" MATCHES "xenial" )
      set( WXP_ARCHIVE https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 )
      list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "-U -f ${WXP_ARCHIVE} wxPython==${WX_VERSION}" )
    else()
      list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "wxPython==${WX_VERSION}" )
    endif()
  else()
    list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "wxPython==${WX_VERSION}" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH AND NOT VIAME_ENABLE_PYTORCH-INTERNAL )
  list( APPEND VIAME_PYTHON_ADV_DEPS pytorch )

  set( PYTORCH_VERSION ${VIAME_PYTORCH_VERSION} )
  set( ARGS_TORCH )
  set( TORCHVISION_STR "" )

  if( VIAME_ENABLE_CUDA )
    set( CUDA_VER_STR "cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}" )
  else()
    set( CUDA_VER_STR "cpu" )
  endif()

  set( PYTORCH_ARCHIVE "https://download.pytorch.org/whl/${CUDA_VER_STR}" )

  if( NOT VIAME_ENABLE_PYTORCH-VIS-INTERNAL )
    if( PYTORCH_VERSION VERSION_GREATER_EQUAL "1.12.0" )
      set( TORCHVISION_STR "torchvision==0.13.0" )
    elseif( PYTORCH_VERSION VERSION_GREATER_EQUAL "1.10.0" )
      set( TORCHVISION_STR "torchvision==0.12.0" )
    elseif( PYTORCH_VERSION VERSION_GREATER_EQUAL "1.9.0" )
      set( TORCHVISION_STR "torchvision==0.10.1" )
    elseif( PYTORCH_VERSION VERSION_GREATER_EQUAL "1.8.2" )
      set( TORCHVISION_STR "torchvision==0.9.2" )
    elseif( PYTORCH_VERSION VERSION_GREATER_EQUAL "1.8.0" )
      set( TORCHVISION_STR "torchvision==0.9.0" )
    elseif( PYTORCH_VERSION VERSION_GREATER_EQUAL "1.7.0" )
      set( TORCHVISION_STR "torchvision==0.8.2" )
    elseif( PYTORCH_VERSION VERSION_GREATER_EQUAL "1.6.0" )
      set( TORCHVISION_STR "torchvision==0.7.0" )
    elseif( PYTORCH_VERSION VERSION_GREATER_EQUAL "1.4.0" )
      set( TORCHVISION_STR "torchvision==0.5.0" )
    else()
      set( TORCHVISION_STR "torchvision" )
    endif()
  endif()

  # Default case
  set( ARGS_TORCH "==${PYTORCH_VERSION} ${TORCHVISION_STR} --extra-index-url ${PYTORCH_ARCHIVE}" )

  # Account for either direct link to package or default case
  string( FIND "${ARGS_TORCH}" "https://" TMP_VAR )
  if( "${TMP_VAR}" EQUAL 0 )
    list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "${ARGS_TORCH}" )
  else()
    list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "torch${ARGS_TORCH}" )
  endif()
endif()

# ------------------------------------- INSTALL ROUTINES -----------------------------------------

if( WIN32 )
  set( EXTRA_INCLUDE_DIRS "${VIAME_INSTALL_PREFIX}/include;$ENV{INCLUDE}" )
  set( EXTRA_LIBRARY_DIRS "${VIAME_INSTALL_PREFIX}/lib;$ENV{LIB}" )

  if( VIAME_ENABLE_PYTHON-INTERNAL )
    set( ENV{PYTHONPATH} "${VIAME_PYTHON_PATH};$ENV{PYTHONPATH}" )
  endif()

  string( REPLACE ";" "----" VIAME_PYTHON_PATH "${VIAME_PYTHON_PATH}" )
  string( REPLACE ";" "----" VIAME_EXECUTABLES_PATH "${VIAME_EXECUTABLES_PATH}" )
  string( REPLACE ";" "----" EXTRA_INCLUDE_DIRS "${EXTRA_INCLUDE_DIRS}" )
  string( REPLACE ";" "----" EXTRA_LIBRARY_DIRS "${EXTRA_LIBRARY_DIRS}" )

  list( APPEND PYTHON_DEP_ENV_VARS "INCLUDE=${EXTRA_INCLUDE_DIRS}" )
  list( APPEND PYTHON_DEP_ENV_VARS "LIB=${EXTRA_LIBRARY_DIRS}" )
  list( APPEND PYTHON_DEP_ENV_VARS "PYTHONIOENCODING=UTF-8" )
else()
  list( APPEND PYTHON_DEP_ENV_VARS "PATH=${VIAME_EXECUTABLES_PATH}" )
  list( APPEND PYTHON_DEP_ENV_VARS "CPPFLAGS=-I${VIAME_INSTALL_PREFIX}/include" )
  list( APPEND PYTHON_DEP_ENV_VARS "LDFLAGS=-L${VIAME_INSTALL_PREFIX}/lib" )
  list( APPEND PYTHON_DEP_ENV_VARS "CC=${CMAKE_C_COMPILER}" )
  list( APPEND PYTHON_DEP_ENV_VARS "CXX=${CMAKE_CXX_COMPILER}" )
endif()

list( APPEND PYTHON_DEP_ENV_VARS "PYTHONPATH=${VIAME_PYTHON_PATH}" )
list( APPEND PYTHON_DEP_ENV_VARS "PYTHONUSERBASE=${VIAME_PYTHON_USERBASE}" )

set( VIAME_PYTHON_DEPS_DEPS fletch )

list( LENGTH VIAME_PYTHON_ADV_DEPS DEP_COUNT )
math( EXPR DEP_COUNT "${DEP_COUNT} - 1" )

foreach( ID RANGE ${DEP_COUNT} )

  list( GET VIAME_PYTHON_ADV_DEPS ${ID} DEP )

  set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} ${DEP} )

  if( "${DEP}" STREQUAL "python-deps" )
    set( PYTHON_LIB_DEPS ${VIAME_PYTHON_DEPS_DEPS} )
    set( CMD ${VIAME_PYTHON_BASIC_DEPS} )
  else()
    set( PYTHON_LIB_DEPS ${VIAME_PYTHON_DEPS_DEPS} python-deps )
    list( GET VIAME_PYTHON_ADV_DEP_CMDS ${ID} CMD )
  endif()

  set( PYTHON_DEP_PIP_CMD pip install --user ${CMD} )
  string( REPLACE " " ";" PYTHON_DEP_PIP_CMD "${PYTHON_DEP_PIP_CMD}" )

  set( PYTHON_DEP_BUILD
    ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
      ${Python_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD}
    )

  if( "${DEP}" STREQUAL "pytorch" )
    set( PYTHON_DEP_INSTALL ${CMAKE_COMMAND}
      -DVIAME_PYTHON_BASE:PATH=${VIAME_PYTHON_INSTALL}
      -DVIAME_PATCH_DIR:PATH=${VIAME_SOURCE_DIR}/packages/patches
      -DVIAME_PYTORCH_VERSION:STRING=${VIAME_PYTORCH_VERSION}
      -P ${VIAME_SOURCE_DIR}/cmake/custom_pytorch_install.cmake )
  elseif( "${DEP}" STREQUAL "python-deps" )
    set( PYTHON_DEP_INSTALL ${CMAKE_COMMAND}
      -DVIAME_PYTHON_BASE:PATH=${VIAME_PYTHON_INSTALL}
      -DVIAME_PATCH_DIR:PATH=${VIAME_SOURCE_DIR}/packages/patches
      -DVIAME_PYTHON_VERSION:STRING=${Python_VERSION}
      -P ${VIAME_SOURCE_DIR}/cmake/custom_python_deps_install.cmake )
  else()
    set( PYTHON_DEP_INSTALL "" )
  endif()

  ExternalProject_Add( ${DEP}
    DEPENDS ${PYTHON_LIB_DEPS}
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${VIAME_CMAKE_DIR}
    USES_TERMINAL_BUILD 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${PYTHON_DEP_BUILD}
    INSTALL_COMMAND "${PYTHON_DEP_INSTALL}"
    INSTALL_DIR ${VIAME_INSTALL_PREFIX}
    LIST_SEPARATOR "----"
    )
endforeach()
