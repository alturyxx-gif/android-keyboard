
# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("voiceinput")

set(SKBUILD ON)

# TODO: Double check this works on devices and doesn't crash if fp16 is unavailable
if(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-fp16 -mfloat-abi=softfp")
endif()

# Custom build: let ggml use the ARMv8.2 dot-product and fp16 instructions, which every
# flagship phone since 2018 has (including the S25 Ultra). This makes quantized speech
# models much faster on the CPU. (Baseline ARMv8 is used otherwise.)
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()

# Custom build: GPU (Vulkan) backend for the speech engine on Android.
# - vulkan.hpp comes from the pinned Khronos Vulkan-Headers submodule (the NDK only ships C headers)
# - spirv.hpp comes from the pinned Khronos SPIRV-Headers submodule
# - the shader compiler (glslc) and libvulkan.so come from the NDK
# The app falls back to the CPU at runtime if the GPU can't be used.
option(VOICE_ENABLE_VULKAN "Build the Vulkan GPU backend for voice input" ON)
if(ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a" AND VOICE_ENABLE_VULKAN)
    set(VK_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include)
    set(SPIRV_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Headers/include)

    set(Vulkan_INCLUDE_DIR ${VK_HEADERS_DIR} CACHE PATH "" FORCE)
    set(Vulkan_GLSLC_EXECUTABLE ${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}/glslc CACHE FILEPATH "" FORCE)
    # ggml needs Vulkan 1.1+ entry points (e.g. vkGetPhysicalDeviceFeatures2), which the NDK
    # only exposes in libvulkan stubs for API 28+. Link the API 33 stub: the real library on
    # the phone (Android 15) has them. Consequence: voice input in this custom build needs
    # Android 9 or newer.
    set(ANDROID_VULKAN_LIB ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-android/33/libvulkan.so)
    set(Vulkan_LIBRARY ${ANDROID_VULKAN_LIB} CACHE FILEPATH "" FORCE)

    # Minimal package config so ggml's find_package(SPIRV-Headers CONFIG) succeeds
    set(SPIRV_SHIM_DIR ${CMAKE_CURRENT_BINARY_DIR}/spirv-headers-shim)
    file(WRITE ${SPIRV_SHIM_DIR}/SPIRV-HeadersConfig.cmake
        "if(NOT TARGET SPIRV-Headers::SPIRV-Headers)\n"
        "  add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)\n"
        "  set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${SPIRV_HEADERS_DIR}\")\n"
        "endif()\n")
    set(SPIRV-Headers_DIR ${SPIRV_SHIM_DIR} CACHE PATH "" FORCE)

    # Vulkan C++ headers must win over the NDK sysroot's older C-only copy
    include_directories(BEFORE ${VK_HEADERS_DIR} ${SPIRV_HEADERS_DIR})

    set(TRANSCRIBE_VULKAN ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(transcribe.cpp)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -g3")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
    # List C/C++ source files with relative paths to this CMakeLists.txt.
    voiceinput.cpp
    jni_common.cpp)

if(ANDROID_ABI STREQUAL armeabi-v7a)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES COMPILE_FLAGS -mfpu=neon)
endif()


target_include_directories(${CMAKE_PROJECT_NAME}
PRIVATE
    transcribe.cpp/include
    transcribe.cpp/ggml/include)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
# The keyboard's own native library (libjni_latinime.so) contains an older copy of ggml
# used by the typing model. Hide every symbol of this library's static dependencies and bind
# all internal references locally, so the two ggml versions can never be mixed up at runtime.
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
    "-Wl,--exclude-libs,ALL"
    "-Wl,-Bsymbolic"
    "-Wl,--gc-sections")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden)

if(ANDROID)
    target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        ggml
        transcribe
        log)
else()
    # Desktop build, used only by the custom build's host tests (JVM unit test loads this
    # library and runs the real WhisperGGML class against a real model and audio file)
    find_package(JNI REQUIRED)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${JNI_INCLUDE_DIRS})
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HOST_TOOL)
    target_link_libraries(${CMAKE_PROJECT_NAME} ggml transcribe)
endif()
