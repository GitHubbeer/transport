cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(CDLS LANGUAGES CXX VERSION 0.0.0)
include(GNUInstallDirs)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

option(CDLS_ENABLE_EXAMPLES "Enable examples" ON)
option(CDLS_ENABLE_OSDI26 "Enable visualization in OSDI '26 paper" OFF)
option(CDLS_ENABLE_TESTS "Enable GETRF scheduling and numerical tests" OFF)
option(CDLS_GETRF_ENABLE_HYBRID "Use BLAS trailing updates for large no-pivot GETRF (OFF for fused-only ablation)" ON)

set(_DISABLE_INSTALLATION ON)

macro(install)
    if (NOT _DISABLE_INSTALLATION)
        _install(${ARGN})
    endif ()
endmacro()

find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA compiler found.")
    set(HAVE_CUDA_COMPILER ON)
else ()
    message(STATUS "CUDA compiler not found.")
    set(HAVE_CUDA_COMPILER OFF)
endif ()

find_package(ROCmCMakeBuildTools QUIET)
if (ROCmCMakeBuildTools_FOUND)
    message(STATUS "AMD ROCM compiler build tools found.")
    set(HAVE_ROCM_COMPILER ON)
else ()
    message(STATUS "AMD ROCM compiler build tools not found.")
    set(HAVE_ROCM_COMPILER OFF)
endif ()

if (HAVE_CUDA_COMPILER)
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)

    set(CUTLASS_ENABLE_LIBRARY OFF)
    set(CUTLASS_ENABLE_PROFILER OFF)
    set(CUTLASS_ENABLE_TESTS OFF)
    set(CUTLASS_ENABLE_EXAMPLES OFF)
    set(CUTLASS_ENABLE_TOOLS OFF)
    set(CUTLASS_ENABLE_HEADERS_ONLY ON)
    add_subdirectory(third_party/cutlass)

    add_subdirectory(src/cuda)
endif ()

if (HAVE_ROCM_COMPILER)
    find_package(hip REQUIRED)

    set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "ROCm GETRF GPU architectures")
    enable_language(HIP)
    find_package(rocblas REQUIRED)
    add_subdirectory(src/rocm)
endif ()

foreach(_getrf_backend cdls_cuda cdls_rocm)
    if(TARGET ${_getrf_backend})
        target_compile_definitions(${_getrf_backend} PRIVATE
            CDLS_GETRF_ENABLE_HYBRID=$<BOOL:${CDLS_GETRF_ENABLE_HYBRID}>)
    endif()
endforeach()

if (CDLS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (HAVE_CUDA_COMPILER)
    target_include_directories(cdls_cuda PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif ()

if (HAVE_ROCM_COMPILER)
    target_include_directories(cdls_rocm PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif ()

if (CDLS_ENABLE_EXAMPLES)
    add_subdirectory(third_party/cxxopts)

    if (HAVE_CUDA_COMPILER)
        add_subdirectory(examples/cuda)
    endif ()

    if (HAVE_ROCM_COMPILER)
        find_package(rocblas REQUIRED)
        find_package(rocsolver REQUIRED)
        find_package(rocthrust REQUIRED)

        add_subdirectory(examples/rocm)
    endif ()
endif ()

add_library(cdls INTERFACE)
target_include_directories(cdls INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if (HAVE_CUDA_COMPILER)
    target_link_libraries(cdls INTERFACE cdls_cuda)
endif ()

if (HAVE_ROCM_COMPILER)
    target_link_libraries(cdls INTERFACE cdls_rocm)
endif ()

set(CDLS_INSTALL_TARGETS cdls)

if (HAVE_CUDA_COMPILER)
    target_compile_options(cdls_cuda PRIVATE
            $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CXX>:-O3>>
    )
    list(APPEND CDLS_INSTALL_TARGETS cdls_cuda)
endif ()

if (HAVE_ROCM_COMPILER)
    target_compile_options(cdls_rocm PRIVATE
            $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CXX>:-O3>>
    )
    list(APPEND CDLS_INSTALL_TARGETS cdls_rocm)
endif ()

set(_DISABLE_INSTALLATION OFF)

install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS ${CDLS_INSTALL_TARGETS}
        EXPORT cdls-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT cdls-targets
        FILE CDLSConfig.cmake
        NAMESPACE CDLS::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CDLS
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/CDLSConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/CDLSConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CDLS"
)
