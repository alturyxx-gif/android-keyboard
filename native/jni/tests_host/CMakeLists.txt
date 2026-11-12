# Host (desktop Linux) tests for the custom build's native typing-model changes.
# They run the REAL bundled model (java/res-large/raw/ml4_q6_k.gguf) on this machine.
#
#   cmake -S native/jni/tests_host -B /tmp/xlm-host -DCMAKE_BUILD_TYPE=Release
#   cmake --build /tmp/xlm-host -j4
#   /tmp/xlm-host/test_scoring  java/res-large/raw/ml4_q6_k.gguf
#   /tmp/xlm-host/test_training java/res-large/raw/ml4_q6_k.gguf
cmake_minimum_required(VERSION 3.22)
project(xlm_host_tests C CXX)

set(CMAKE_CXX_STANDARD 17)
set(EXT_BUILD ON CACHE BOOL "" FORCE)
set(ANDROID_ABI "host")
find_package(JNI REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_BINARY_DIR}/latinime)

set(JNI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include(${JNI_ROOT}/NativeFileList.cmake)
set(HOST_INCLUDES
    ${JNI_ROOT}
    ${JNI_ROOT}/src
    ${JNI_ROOT}/src/sentencepiece/builtin_pb
    ${JNI_ROOT}/src/third_party
    ${JNI_ROOT}/src/third_party/protobuf-lite
    ${JNI_ROOT}/src/third_party/esaxx
    ${JNI_ROOT}/src/third_party/darts_clone
    ${JNI_ROOT}/src/third_party/absl
    ${JNI_INCLUDE_DIRS})

set(EXTRA_test_scoring "")
set(EXTRA_test_training score_helper.cpp)
foreach(t test_scoring test_training)
    add_executable(${t} ${t}.cpp ${EXTRA_${t}} host_stubs.cpp ${JNI_ROOT}/src/jni_utils.cpp)
    target_include_directories(${t} PRIVATE ${HOST_INCLUDES})
    target_compile_options(${t} PRIVATE -w -include cstring)
    target_link_libraries(${t} jni_latinime_common_static pthread)
endforeach()

# Experiment driver for tuning personalisation training (not a test)
add_executable(exp_train exp_train.cpp score_helper.cpp host_stubs.cpp ${JNI_ROOT}/src/jni_utils.cpp)
target_include_directories(exp_train PRIVATE ${HOST_INCLUDES})
target_compile_options(exp_train PRIVATE -w -include cstring)
target_link_libraries(exp_train jni_latinime_common_static pthread)
