cmake_minimum_required(VERSION 3.20)
project(ATPA CXX)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(PROJECT_VERSION 1.0)

macro(add_target_property _target _prop _value)
   get_target_property(_oldprop "${_target}" ${_prop})
   if(NOT _oldprop)
      set_target_properties("${_target}" PROPERTIES ${_prop} "${_value}")
   else(NOT _oldprop)
      set_target_properties("${_target}" PROPERTIES ${_prop} "${_oldprop} ${_value}")
   endif(NOT _oldprop)
endmacro(add_target_property)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

# by default build optimized code with debug symbols
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif ()

# in DEBUG mode make verbose Makefile
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g -DDEBUG -D__DEBUG -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG -D__DEBUG -Wall")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ftree-vectorize -ffast-math -DNODEBUG")
message(STATUS "Using CXX flags for ${CMAKE_BUILD_TYPE}: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

find_package(AnalysisTree REQUIRED)
find_package(ROOT REQUIRED RIO)

if (ROOT_FOUND)
    message(STATUS "Using ROOT: ${ROOT_VERSION} <${ROOT_CONFIG}>")
    include_directories(${CMAKE_SOURCE_DIR} ${ROOT_INCLUDE_DIR} ${ROOT_INCLUDE_DIRS})
    include_directories($ENV{OnnxRuntime_DIR}/include)
    include(${ROOT_USE_FILE})
endif (ROOT_FOUND)

include_directories(${CMAKE_SOURCE_DIR} ${AnalysisTree_INCLUDE_DIR})

#************
set(SOURCES
  ATreePredictionAdder.cpp
  ONNXConfigParser.cpp
  ONNXConfigManager.cpp
  ONNXSingleBinModel.cpp
  ONNXRunner.cpp
  )

string(REPLACE ".cpp" ".hpp" HEADERS "${SOURCES}")
add_library(ATPA SHARED ${SOURCES} G__ATPA.cxx)
        
if(ROOT_FOUND)
  ROOT_GENERATE_DICTIONARY(G__ATPA ${HEADERS} LINKDEF ATPALinkDef.h OPTIONS "-DDO_TPCCATRACKER_EFF_PERFORMANCE")
  include_directories(${CMAKE_SOURCE_DIR})
  target_link_libraries(ATPA ${ROOT_LIBRARIES} AnalysisTreeBase AnalysisTreeInfra)
endif(ROOT_FOUND)  
add_target_property(ATPA COMPILE_FLAGS "-DDO_TPCCATRACKER_EFF_PERFORMANCE")

#BEGIN ONNX
add_library(libonnxruntime STATIC IMPORTED)
# Provide the full path to the library, so CMake knows where to find it.
set_target_properties(libonnxruntime PROPERTIES IMPORTED_LOCATION $ENV{OnnxRuntime_DIR}/lib/libonnxruntime.so)
target_link_libraries(ATPA libonnxruntime)

find_package(Boost 1.78 REQUIRED COMPONENTS json)
target_link_libraries(ATPA Boost::boost Boost::json)
#END ONNX

add_executable(at_tree_prediction_adder main.cpp)
add_dependencies(at_tree_prediction_adder ATPA)
add_target_property(at_tree_prediction_adder COMPILE_FLAGS "-DDO_TPCCATRACKER_EFF_PERFORMANCE")
target_link_libraries(at_tree_prediction_adder ATPA)

install(TARGETS ATPA EXPORT ATPATargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

install(
        FILES
        ${HEADERS}
        DESTINATION
        include
        COMPONENT
        Devel
)

set(PCM_FILE_NAME libATPA)

install(
        FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PCM_FILE_NAME}_rdict.pcm"
        DESTINATION
        lib
        OPTIONAL
)

install (TARGETS at_tree_prediction_adder RUNTIME DESTINATION bin)
#************

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/ATPA/ATPAConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

export(EXPORT ATPATargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/ATPA/ATPATargets.cmake")

message(STATUS "ATPATargets: ${ATPATargets}")
        
set(ConfigPackageLocation lib/cmake/ATPA)

set(ATPA_INCLUDE_DIR "include")
set(ATPA_LIBRARY_DIR "lib")

configure_package_config_file(
        ATPAConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/ATPA/ATPAConfig.cmake
        INSTALL_DESTINATION ${ConfigPackageLocation}
        PATH_VARS ATPA_INCLUDE_DIR ATPA_LIBRARY_DIR
)

install(EXPORT ATPATargets
        FILE
        ATPATargets.cmake
        DESTINATION
        ${ConfigPackageLocation}
        )

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ATPA/ATPAConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ATPA/ATPAConfigVersion.cmake"
        DESTINATION
        ${ConfigPackageLocation}
        COMPONENT
        Devel
        )
