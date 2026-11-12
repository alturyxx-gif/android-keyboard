#include <cstdlib>
/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "jni_common.h"
#include "defines.h"
#include "voiceinput.h"

/*
 * Returns the JNI version on success, -1 on failure.
 */
jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env = nullptr;

    // Custom build: on the Adreno 830 (S25 Ultra) the Vulkan backend's fp16 path produced
    // garbage encoder output with Parakeet ("decoder: 22 iters, 0 tokens"). Force full fp32
    // precision on the GPU. Must be set before the Vulkan device is first initialised.
    setenv("GGML_VK_DISABLE_F16", "1", 0);

    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        AKLOGE("ERROR: GetEnv failed");
        return -1;
    }
    ASSERT(env);
    if (!env) {
        AKLOGE("ERROR: JNIEnv is invalid");
        return -1;
    }
    if (!register_WhisperGGML(env)) {
        AKLOGE("ERROR: Failed to register WhisperGGML");
        return -1;
    }
    /* success -- return valid version number */
    return JNI_VERSION_1_6;
}

int registerNativeMethods(JNIEnv *env, const char *const className, const JNINativeMethod *methods,
                          const int numMethods) {
    jclass clazz = env->FindClass(className);
    if (!clazz) {
        AKLOGE("Native registration unable to find class '%s'", className);
        return JNI_FALSE;
    }
    if (env->RegisterNatives(clazz, methods, numMethods) != 0) {
        AKLOGE("RegisterNatives failed for '%s'", className);
        env->DeleteLocalRef(clazz);
        return JNI_FALSE;
    }
    env->DeleteLocalRef(clazz);
    return JNI_TRUE;
}