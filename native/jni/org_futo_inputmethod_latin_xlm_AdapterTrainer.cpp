//
// Created by alex on 11/7/23.
//

#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "org_futo_inputmethod_latin_xlm_AdapterTrainer.h"
#include "defines.h"
#include "jni_common.h"
#include "ggml/finetune.h"
#include "sentencepiece/sentencepiece_processor.h"
#include "jni_utils.h"
#include "ggml/ModelMeta.h"

namespace latinime {
    struct AdapterTrainerState {
        std::string baseModelPath;
        std::string loraCachePath;
        std::string outputModelPath;
        float outputScale;

        ModelMetadata metadata;

        sentencepiece::SentencePieceProcessor spm;
        struct train_params params;

        static void OnLossCallback(void *userdata, float loss) {
            auto *state = reinterpret_cast<AdapterTrainerState *>(userdata);
            state->OnLoss(loss);
        }

        static void OnProgressCallback(void *userdata, float progress) {
            auto *state = reinterpret_cast<AdapterTrainerState *>(userdata);
            state->OnProgress(progress);
        }

        JNIEnv *env;
        jobject callbackObject;
        jmethodID lossMethodId;
        jmethodID progressMethodId;
        void OnLoss(float loss) const {
            env->CallVoidMethod(callbackObject, lossMethodId, loss);
        }

        void OnProgress(float progress) const {
            env->CallVoidMethod(callbackObject, progressMethodId, progress);
        }

        bool Initialize() {
            metadata = loadModelMetadata(baseModelPath.c_str());

            // TODO: Gracefully handle errors
            ASSERT(!metadata.error);
            ASSERT(metadata.ext_tokenizer_type == ExternalTokenizerType::SentencePiece);


            params = get_default_train_params();
            params.common.fn_train_data = "";
            params.common.fn_checkpoint_in = "";
            params.common.fn_checkpoint_out = "";
            params.fn_model_base = baseModelPath.c_str();
            params.fn_lora_out = loraCachePath.c_str();

            params.common.fill_with_next_samples = true;
            params.common.n_threads = 6;
            params.common.n_gradient_accumulation = 2;
            params.common.n_batch = 2;
            params.common.n_ctx = 64;
            // (the custom build overrides these via SetPlan before training)
            params.common.sample_random_offsets = true;

            params.common.warmup = 10;
            params.common.n_epochs = 1;
            params.common.adam_alpha = 1e-3;
            params.common.adam_n_iter = 128;

            // Increasing/decreasing this doesn't appear to significantly affect training time
            params.lora_r = 16;
            params.lora_alpha = 16;

            params.common.callbacks.userdata = this;
            params.common.callbacks.loss     = AdapterTrainerState::OnLossCallback;
            params.common.callbacks.progress = AdapterTrainerState::OnProgressCallback;

            if(!spm.LoadFromSerializedProto(metadata.ext_tokenizer_data).ok()){
                AKLOGE("Failed to load tokenizer!");
                return false;
            }

            return true;
        }

        void AddTrainingExample(const std::string &example) {
            std::vector<llama_token> result = spm.EncodeAsIds(example);
            params.training_data.push_back(result);
        }

        int Train() const {
            return finetune_train(params);
        }

        // Stage 2.5: training size scales with how much history there is
        void SetPlan(int steps, int ctx, int batch, int threads) {
            if(steps > 0) params.common.adam_n_iter = steps;
            if(ctx > 0) params.common.n_ctx = ctx;
            if(batch > 0) params.common.n_batch = batch;
            if(threads > 0) params.common.n_threads = threads;
            params.common.warmup = std::max(10, params.common.adam_n_iter / 20);
            // The original code stopped after 1 epoch, which with a few hundred examples is only
            // ~15 optimiser steps - far too few to learn anything. Let the step count decide.
            params.common.n_epochs = -1;
        }

        void UpdateHistoryAndCount(std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end) {
            std::chrono::duration<double> elapsed_seconds = end - start;

            int num_examples = params.training_data.size();
            int num_tokens = 0;
            for(const auto & example: params.training_data) {
                num_tokens += example.size();
            }

            time_t rawtime;
            struct tm * timeinfo;
            char date_time[32];

            // Convert time_point to time_t
            rawtime = std::chrono::system_clock::to_time_t(start);
            // Convert time_t to tm struct
            timeinfo = localtime(&rawtime);

            // Format the date and time in ISO format
            strftime(date_time, sizeof(date_time), "%Y-%m-%d %H:%M:%SZ", timeinfo);

            // Create a stringstream object
            std::stringstream ss;

            // Format the string using the stringstream object
            ss << "\n" << date_time << ": Fine-tuned on " << num_examples << " examples (" << num_tokens << " tokens), took "
               << std::fixed << std::setprecision(2) << elapsed_seconds.count() / 60.0 << " minutes";

            // Convert the stringstream object to a std::string
            std::string result = ss.str();

            metadata.finetuning_count += 1;
            metadata.history.append(result);
        }
    };

    static jlong xlm_AdapterTrainer_open(JNIEnv *env, jclass clazz, jstring baseModelPathStr, jstring loraCacheStr, jstring outputModelPathStr, float outputScale) {
        auto *state = new AdapterTrainerState();
        state->baseModelPath   = jstring2string(env, baseModelPathStr);
        state->loraCachePath   = jstring2string(env, loraCacheStr);
        state->outputModelPath = jstring2string(env, outputModelPathStr);
        state->outputScale = outputScale;

        state->env = env;

        if(!state->Initialize()) {
            delete state;
            return 0;
        }

        return reinterpret_cast<jlong>(state);
    }

    static void xlm_AdapterTrainer_close(JNIEnv *env, jclass clazz, jlong statePtr) {
        auto *state = reinterpret_cast<AdapterTrainerState *>(statePtr);
        if(state == nullptr) return;
        delete state;
    }

    static void xlm_AdapterTrainer_addExample(JNIEnv *env, jclass clazz, jlong statePtr, jstring exampleStr) {
        auto *state = reinterpret_cast<AdapterTrainerState *>(statePtr);
        state->AddTrainingExample(jstring2string(env, exampleStr));
    }

    // TODO: Callback for progress
    // Trains the personalisation LoRA adapter and writes it to loraCachePath.
    // Unlike FUTO's original version, the adapter is NOT merged into the model file:
    // it stays separate so it can be rolled back, reset or weakened (Stage 2.2).
    static jboolean xlm_AdapterTrainer_train(JNIEnv *env, jobject instance, jlong statePtr) {

        jclass clazz = env->GetObjectClass(instance);
        ASSERT(clazz);

        jmethodID progressMethodId = env->GetMethodID(clazz, "emitProgress", "(F)V");
        jmethodID lossMethodId = env->GetMethodID(clazz, "emitLoss", "(F)V");
        ASSERT(progressMethodId);
        ASSERT(lossMethodId);

        auto *state = reinterpret_cast<AdapterTrainerState *>(statePtr);
        state->env = env;
        state->lossMethodId = lossMethodId;
        state->progressMethodId = progressMethodId;
        state->callbackObject = instance;

        int result = state->Train();
        if(result != 0) {
            AKLOGE("train returned with non-zero code %d", result);
            return JNI_FALSE;
        }

        return JNI_TRUE;
    }

    static void xlm_AdapterTrainer_setPlan(JNIEnv *env, jclass clazz, jlong statePtr, jint steps, jint ctx, jint batch, jint threads) {
        auto *state = reinterpret_cast<AdapterTrainerState *>(statePtr);
        if(state == nullptr) return;
        state->SetPlan(steps, ctx, batch, threads);
    }

    static const JNINativeMethod sMethods[] = {
            {
                    const_cast<char *>("openNative"),
                    const_cast<char *>("(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;F)J"),
                    reinterpret_cast<void *>(xlm_AdapterTrainer_open)
            },
            {
                    const_cast<char *>("closeNative"),
                    const_cast<char *>("(J)V"),
                    reinterpret_cast<void *>(xlm_AdapterTrainer_close)
            },
            {
                    const_cast<char *>("addExample"),
                    const_cast<char *>("(JLjava/lang/String;)V"),
                    reinterpret_cast<void *>(xlm_AdapterTrainer_addExample)
            },
            {
                    const_cast<char *>("train"),
                    const_cast<char *>("(J)Z"),
                    reinterpret_cast<void *>(xlm_AdapterTrainer_train)
            },
            {
                    const_cast<char *>("setPlan"),
                    const_cast<char *>("(JIIII)V"),
                    reinterpret_cast<void *>(xlm_AdapterTrainer_setPlan)
            },

    };

    int register_AdapterTrainer(JNIEnv *env) {
        const char *const kClassPathName = "org/futo/inputmethod/latin/xlm/AdapterTrainer";
        return registerNativeMethods(env, kClassPathName, sMethods, NELEMS(sMethods));
    }
}