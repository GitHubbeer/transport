# Use the same pinned CK headers as the optimized upstream GETRF. A caller can
# supply a checkout to avoid fetching it; no sibling-project paths are required.
set(CDLS_CK_SOURCE_DIR "" CACHE PATH "Composable Kernel source tree (ROCm 6.4.2)")
if(NOT CDLS_CK_SOURCE_DIR)
    include(FetchContent)
    FetchContent_Declare(cdls_ck
        GIT_REPOSITORY https://github.com/ROCm/composable_kernel.git
        GIT_TAG a8c5bd9b9ad950c3e742877e01cb784da91664e3)
    FetchContent_GetProperties(cdls_ck)
    if(NOT cdls_ck_POPULATED)
        FetchContent_Populate(cdls_ck)
    endif()
    set(CDLS_CK_SOURCE_DIR "${cdls_ck_SOURCE_DIR}")
endif()
if(NOT EXISTS "${CDLS_CK_SOURCE_DIR}/include/ck/config.h.in")
    message(FATAL_ERROR "CDLS_CK_SOURCE_DIR must point to a CK source checkout")
endif()
foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES)
    if(NOT arch MATCHES "^(gfx90a|gfx940|gfx941|gfx942)$")
        message(FATAL_ERROR "FP64 CK GETRF requires wave64 gfx90a/gfx94x; got ${arch}")
    endif()
endforeach()
set(CK_ENABLE_FP64 ON)
set(CK_USE_XDL ON)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/ck")
configure_file("${CDLS_CK_SOURCE_DIR}/include/ck/config.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/ck/config.h")

add_library(cdls_rocm SHARED common.cpp getrf.cpp)
set_source_files_properties(getrf.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(cdls_rocm PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_compile_features(cdls_rocm PRIVATE cxx_std_17)
target_include_directories(cdls_rocm PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_include_directories(cdls_rocm SYSTEM PRIVATE "${CDLS_CK_SOURCE_DIR}/include")
target_link_libraries(cdls_rocm PRIVATE hip::host roc::rocblas)
