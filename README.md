
<img src="http://www.viametoolkit.org/wp-content/uploads/2016/08/viami_logo.png" alt="VIAME Logo" width="200" height="78">

VIAME is a computer vision application designed for do-it-yourself artificial intelligence including
object detection, object tracking, image/video annotation, image/video search, image mosaicing,
image enhancement, size measurement, rapid model generation, and tools for the evaluation of
different algorithms. Originally targetting marine species analytics, VIAME now contains many
common algorithms and libraries, and is also useful as a generic computer vision toolkit. It
contains a number of standalone tools for accomplishing the above, a pipeline framework which
can connect C/C++, python, and matlab nodes together in a multi-threaded fashion, and multiple
algorithms resting on top of the pipeline infrastructure. Lastly, both desktop and web user
interfaces exist for deployments in different types of environments, with an open annotation
archive and example of the web platform available at [viame.kitware.com](https://viame.kitware.com).


Documentation
-------------

The [User's Quick-Start Guide](https://data.kitware.com/api/v1/item/5fdaf1dd2fa25629b99843f8/download),
[Tutorial Videos](https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw), 
and [Developer's Manual](http://viame.readthedocs.io/en/latest/) are more comprehensive,
but select entries are also listed below broken down by individual functionality:


[Documentation Overview](https://viame.readthedocs.io/en/latest/section_links/documentation_overview.html) <>
[Install or Build Instructions](examples/building_and_installing_viame) <>
[All Examples](https://github.com/Kitware/VIAME/tree/master/examples) <>
[DIVE Interface](https://kitware.github.io/dive) <>
[VIEW Interface](examples/annotation_and_visualization) <>
[Search and Rapid Model Generation](examples/search_and_rapid_model_generation) <>
[Object Detector CLI](examples/object_detection) <>
[Object Tracker CLI](examples/object_tracking) <>
[Detector Training CLI](examples/object_detector_training) <>
[Evaluation of Detectors](examples/scoring_and_roc_generation) <>
[Detection File Formats](https://viame.readthedocs.io/en/latest/section_links/detection_file_conversions.html) <>
[Calibration and Image Enhancement](examples/image_enhancement) <>
[Registration and Mosaicing](examples/image_registration)  <>
[Stereo Measurement and Depth Maps](examples/measurement_using_stereo) <>
[Pipelining Overview](https://github.com/Kitware/kwiver) <>
[Core Class and Pipeline Info](http://kwiver.readthedocs.io/en/latest/architecture.html) <>
[Plugin Integration](examples/hello_world_pipeline) <>
[Example Plugin Templates](plugins/templates) <>
[Embedding Algorithms in C++](examples/using_detectors_in_cxx_code)

Installations
-------------

For a full installation guide and description of the various flavors of VIAME, see the
quick-start guide, above. The full desktop version is provided as either a .msi, .zip or
.tar file. Alternatively, standalone annotators (without any processing algorithms)
are available via smaller installers. Lastly, docker files are available for both VIAME
Desktop and Web (below). For full desktop installs, extract the binaries and place them
in a directory of your choosing, for example /opt/noaa/viame on Linux
or C:\Program Files\VIAME on Windows. If using packages built with GPU support, make sure
to have sufficient video drivers installed, version 465.19 or higher. The best way to
install drivers depends on your operating system. This isn't required if just using
manual annotators (or frame classifiers only). The binaries are quite large,
in terms of disk space, due to the inclusion of multiple default model files and
programs, but if just building your desired features from source (e.g. for embedded
apps) they are much smaller.

**Installation Requirements:** <br>
* Up to 8 Gb of Disk Space for the Full Installation <br>
* Windows 7\*, 8, 10, or 11 (64-Bit) or Linux (64-Bit, e.g. RHEL, CentOS, Ubuntu) <br>
  * Windows 7 requires some updates and service packs installed, e.g. [KB2533623](https://www.microsoft.com/en-us/download/details.aspx?id=26764). <br>
  * MacOS is currently only supported running standalone annotation tools, see below.

**Installation Recommendations:** <br>
* NVIDIA Drivers (Version 465.19 or above, 
Windows 
[\[1\]](https://www.nvidia.com/Download/index.aspx?lang=en-us)
[\[2\]](https://developer.nvidia.com/cuda-downloads)
Ubuntu 
[\[1\]](https://linuxhint.com/ubuntu_nvidia_ppa/)
[\[2\]](https://developer.nvidia.com/cuda-downloads)
CentOS 
[\[1\]](https://developer.nvidia.com/cuda-downloads)
[\[2\]](https://www.nvidia.com/Download/index.aspx?lang=en-us)) <br>
* A [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) with 8 Gb or more VRAM <br>

**Windows Full Desktop Binaries:** <br>
* VIAME v0.20.0 Windows, GPU Enabled, Wizard (.msi) (Coming Soon...) <br>
* [VIAME v0.20.0 Windows, GPU Enabled, Mirror1 (.zip)](https://drive.google.com/file/d/1Jdu4oYtPu7rOPnNABgaLez--9lv5ciHd/view?usp=share_link) <br>
* [VIAME v0.20.0 Windows, GPU Enabled, Mirror2 (.zip)](https://data.kitware.com/api/v1/item/63e3f5ff046e924d0df1989a/download) <br>
* [VIAME v0.20.0 Windows, CPU Only, Mirror1 (.zip)](https://drive.google.com/file/d/1862S57V_fBVgn1tX4tEUPnRz2jj6LI9r/view?usp=share_link) <br>
* [VIAME v0.20.0 Windows, CPU Only, Mirror2 (.zip)](https://data.kitware.com/api/v1/item/63e5b6777b0dfcc98f6670d7/download)

**Linux Full Desktop Binaries:** <br>
* [VIAME v0.20.0 Linux, GPU Enabled, Mirror1 (.tar.gz)](https://drive.google.com/file/d/1R1wQ6x1pp6h8oo71VUcQiEaLZh5nfmgr/view?usp=share_link) <br>
* [VIAME v0.20.0 Linux, GPU Enabled, Mirror2 (.tar.gz)](https://data.kitware.com/api/v1/item/645a7a0458fefcae81a9f728/download) <br>
* [VIAME v0.20.0 Linux, CPU Only, Mirror1 (.tar.gz)](https://drive.google.com/file/d/14ONeYjuK_7oyeZ50G8c9v45sKm6vCieg/view?usp=share_link) <br>
* [VIAME v0.20.0 Linux, CPU Only, Mirror2 (.tar.gz)](https://data.kitware.com/api/v1/item/63e5b5ef7b0dfcc98f6670d3/download)

**Web Applications**: <br>
* [VIAME Online Web Annotator and Public Annotation Archive](https://viame.kitware.com/) <br>
* [VIAME Web Local Installation Instructions](https://kitware.github.io/dive/Deployment-Overview/) <br>
* [VIAME Web Source Repository](https://github.com/Kitware/dive)

**DIVE Standalone Desktop Annotator:** <br>
* [DIVE Installers (Linux, Mac, Windows)](https://github.com/Kitware/dive/releases)

**SEAL Standalone Desktop Annotator:** <br>
* [SEAL Windows 7/8/10, GPU Enabled (.zip)](https://data.kitware.com/api/v1/item/602296172fa25629b95482f6/download) <br>
* [SEAL Windows 7/8/10, CPU Only (.zip)](https://data.kitware.com/api/v1/item/602295642fa25629b9548196/download) <br>
* [SEAL CentOS 7, GPU Enabled (.tar.gz)](https://data.kitware.com/api/v1/item/6023362a2fa25629b957c365/download) <br>
* [SEAL Generic Linux, GPU Enabled (.tar.gz)](https://data.kitware.com/api/v1/item/6023359c2fa25629b957c2f3/download)

**Optional Add-Ons and Model Files:** <br>
* [Arctic Seals Models, Windows](https://data.kitware.com/api/v1/item/5e30b8ffaf2e2eed3545bff6/download) <br>
* [Arctic Seals Models, Linux](https://data.kitware.com/api/v1/item/5e30b283af2e2eed3545a888/download) <br>
* [EM Tuna Detectors, All OS](https://viame.kitware.com/api/v1/item/627b326cc4da86e2cd3abb5b/download) <br>
* [HabCam Models (Scallop, Skate, Flatfish), Linux](https://viame.kitware.com/api/v1/item/645a7f6d4c19222431de7953/download) <br>
* [Motion Detector Model, All OS](https://viame.kitware.com/api/v1/item/627b326fea630db5587b577b/download) <br>
* [MOUSS Deep 7 Bottomfish Models, All OS](https://viame.kitware.com/api/v1/item/627b3282c4da86e2cd3abb5d/download) <br>
* [Penguin Head FF Models, All OS](https://viame.kitware.com/api/v1/item/627b3289ea630db5587b577d/download) <br>
* [Sea Lion Models, All OS](https://viame.kitware.com/api/v1/item/641d644abfe642e7e4f54655/download) <br>
* [SEFSC 100-200 Class Fish Models, All OS](https://viame.kitware.com/api/v1/item/627b32b1994809b024f207a7/download)

Note: To install Add-Ons and Patches, copy them into an existing VIAME installation folder.
Folders should match, for example, the Add-On packages contains a 'configs' folder, and the
main installation also contains a 'configs' folder so they should just be merged.


Docker Images
-------------

Docker images are available on: https://hub.docker.com. For a default container with just core
algorithms, runnable via command-line, see:

kitware/viame:gpu-algorithms-latest

This image is headless (ie, it contains no GUI) and contains a VIAME desktop (not web)
installation in the folder /opt/noaa/viame. For links to the VIAME-Web docker containers see the
above section in the installation documentation. Most add-on models are not included in the
instance but can be downloaded via running the script download_viame_addons.sh in the bin folder.

Quick Build Instructions
------------------------

These instructions are intended for developers or those interested in building the latest master
branch. More in-depth build instructions can be found [here](examples/building_and_installing_viame),
but the software can be built either as a super-build, which builds most of its dependencies
alongside itself, or standalone. To build VIAME requires, at a minimum, [Git](https://git-scm.com/),
[CMake](https://cmake.org/), and a [C++ compiler](http://www.cplusplus.com/doc/tutorial/introduction/).
Installing Python and CUDA is also recommended. If using CUDA, versions 11.7 or 11.6 are
preferred, with CUDNN 8. Other CUDA or CUDNN versions may or may not work. For python distributions,
at a minimum Python3.6 or above, Pip, and Numpy is necessary.

To build on the command line in Linux, use the following commands, only replacing [source-directory]
and [build-directory] with locations of your choice. While these directories can be the same,
it's good practice to have a 'src' checkout then a seperate 'build' directory alongside it:

	git clone https://github.com/VIAME/VIAME.git [source-directory]

	cd [source-directory] && git submodule update --init --recursive

Next, create a build directory and run the following `cmake` command (or alternatively
use the cmake GUI if you are not using the command line interface):

	mkdir [build-directory] && cd [build-directory]

	cmake -DCMAKE_BUILD_TYPE:STRING=Release [source-directory]

Once your `cmake` command has completed, you can configure any build flags you want
using 'ccmake' or the cmake GUI, and then build with the following command on Linux:

	make -j8

Or alternatively by building it in Visual Studio or your compiler of choice on
Windows. On Linux, '-j8' tells the build to run multi-threaded using 8 threads, this
is useful for a faster build though if you get an error it can be difficult to see
it, in which case running just 'make' might be more helpful. For Windows,
currently VS2019 is the most tested compiler.

There are several optional arguments to viame which control which plugins get built,
such as those listed below. If a plugin is enabled that depends on another dependency
such as OpenCV) then the dependency flag will be forced to on. If uncertain what to turn
on, it's best to just leave the default enable and disable flags which will build most
(though not all) functionalities. These are core components we recommend leaving turned on:


<center>

| Flag                         | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| VIAME_ENABLE_OPENCV          | Builds OpenCV and basic OpenCV processes (video readers, simple GUIs)          |
| VIAME_ENABLE_VXL             | Builds VXL and basic VXL processes (video readers, image filters)              |
| VIAME_ENABLE_PYTHON          | Turns on support for using python processes (multiple algorithms)              |
| VIAME_ENABLE_PYTORCH         | Installs all pytorch processes (detectors, trackers, classifiers)              |

</center>


And a number of flags which control which system utilities and optimizations are built, e.g.:


<center>

| Flag                         | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| VIAME_ENABLE_CUDA            | Enables CUDA (GPU) optimizations across all packages                           |
| VIAME_ENABLE_CUDNN           | Enables CUDNN (GPU) optimizations across all processes                         |
| VIAME_ENABLE_DIVE            | Enables DIVE GUI (annotation and training on multiple sequences)               |
| VIAME_ENABLE_VIVIA           | Builds VIVIA GUIs (VIEW and SEARCH for annotation and video search)            |
| VIAME_ENABLE_KWANT           | Builds KWANT detection and track evaluation (scoring) tools                    |
| VIAME_ENABLE_DOCS            | Builds Doxygen class-level documentation (puts in install tree)                |
| VIAME_BUILD_DEPENDENCIES     | Build VIAME as a super-build, building all dependencies (default)              |
| VIAME_INSTALL_EXAMPLES       | Installs examples for the above modules into install/examples tree             |
| VIAME_DOWNLOAD_MODELS        | Downloads pre-trained models for use with the examples and interfaces          |

</center>


And lastly, a number of flags which build algorithms or interfaces with more specialized functionality:


<center>

| Flag                         | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| VIAME_ENABLE_TENSORFLOW      | Builds TensorFlow object detector plugin                                       |
| VIAME_ENABLE_DARKNET         | Builds Darknet (YOLO) object detector plugin                                   |
| VIAME_ENABLE_TENSORRT        | Builds TensorRT object detector plugin                                         |
| VIAME_ENABLE_BURNOUT         | Builds Burn-Out based pixel classifier plugin                                  |
| VIAME_ENABLE_SMQTK           | Builds SMQTK plugins to support image/video indexing and search                |
| VIAME_ENABLE_SCALLOP_TK      | Builds Scallop-TK based object detector plugin                                 |
| VIAME_ENABLE_SEAL            | Builds Seal multi-modality GUI                                                 |
| VIAME_ENABLE_ITK             | Builds ITK cross-modality image registration                                   |
| VIAME_ENABLE_UW_CLASSIFIER   | Builds UW fish classifier plugin                                               |
| VIAME_ENABLE_MATLAB          | Turns on support for and installs all matlab processes                         |
| VIAME_ENABLE_LANL            | Builds an additional (Matlab) scallop detector                                 |

</center>


Source Code Layout
------------------
<pre>
 VIAME
   ├── cmake               # CMake configuration files for subpackages
   ├── docs                # Documentation files and manual (pre-compilation)
   ├── configs             # All system-runnable config files and models
   │   ├── pipelines       # All processing pipeline configs
   │   │   └── models      # All models, which only get downloaded based on flags
   │   ├── prj-linux       # Default linux project files
   │   └── prj-windows     # Default windows project files 
   ├── examples            # All runnable examples and example tutorials
   ├── packages            # External projects used by the system
   │   ├── kwiver          # Processing backend infastructure
   │   ├── fletch          # Dependency builder for things which don't change often
   │   ├── kwant           # Scoring and detector evaluation tools
   │   ├── vivia           # Baseline desktop GUIs (v1.0)
   │   └── ...             # Assorted other packages (typically for algorithms)
   ├── plugins             # Integrated algorithms or wrappers around external projects
   │   └── ...             # Assorted plugins (detectors, depth maps, filters, etc.)
   ├── tools               # Standalone tools or scripts, often building on the above
   └── README.md           # Project introduction page that you are reading
   └── RELEASE_NOTES.md    # A list of the latest updates in the system per version
</pre>


Update Instructions
-------------------

If you already have a checkout of VIAME and want to switch branches or
update your code, it is important to re-run:

	git submodule update --init --recursive

After switching branches to ensure that you have on the correct hashes
of sub-packages within the build. Very rarely you may also need to run:

	git submodule sync

Just in case the address of submodules has changed. You only need to
run this command if you get a "cannot fetch hash #hashid" error.


License, Citations, and Acknowledgements
----------------------------------------

VIAME is released under a BSD-3 license.

A non-exhaustive list of relevant papers used within the project alongside contributors
can be found [here](docs/citations.md).

VIAME was developed with funding from multiple sources, with special thanks
to those listed [here](docs/acknowledgements.md).
