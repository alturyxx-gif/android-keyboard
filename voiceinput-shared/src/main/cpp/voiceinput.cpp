#include <string>
#include <algorithm>
#include <cstring>
#include <functional>
#include <sys/stat.h>
#include <atomic>
#include <deque>
#include <mutex>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <map>
#include <jni.h>
#include <unistd.h>
#include "defines.h"
#include "voiceinput.h"
#include "jni_common.h"
#include "transcribe.h"
#include "transcribe/whisper.h"
#include "transcribe/parakeet.h"
#include "transcribe/asr4all.h"

std::string jstring2string(JNIEnv *env, jstring jStr) {
    if(jStr == nullptr) {
        AKLOGE("jstring is null!");
        return "";
    }

    const jsize stringUtf8Length = env->GetStringUTFLength(jStr);
    if (stringUtf8Length <= 0) {
        return "";
    }
    char stringChars[stringUtf8Length + 1];
    env->GetStringUTFRegion(jStr, 0, env->GetStringLength(jStr), stringChars);
    stringChars[stringUtf8Length] = '\0';

    return {stringChars};
}

jstring string2jstring(JNIEnv *env, const char *str) {
    jobject bb = env->NewDirectByteBuffer((void *)str, strlen(str));

    jclass cls_Charset = env->FindClass("java/nio/charset/Charset");
    jmethodID mid_Charset_forName = env->GetStaticMethodID(cls_Charset, "forName", "(Ljava/lang/String;)Ljava/nio/charset/Charset;");
    jobject charset = env->CallStaticObjectMethod(cls_Charset, mid_Charset_forName, env->NewStringUTF("UTF-8"));

    jmethodID mid_Charset_decode = env->GetMethodID(cls_Charset, "decode", "(Ljava/nio/ByteBuffer;)Ljava/nio/CharBuffer;");
    jobject cb = env->CallObjectMethod(charset, mid_Charset_decode, bb);
    env->DeleteLocalRef(bb);

    jclass cls_CharBuffer = env->FindClass("java/nio/CharBuffer");
    jmethodID mid_CharBuffer_toString = env->GetMethodID(cls_CharBuffer, "toString", "()Ljava/lang/String;");
    jstring s = (jstring)env->CallObjectMethod(cb, mid_CharBuffer_toString);

    return s;
}

struct whisper_context {
    int x;
};

struct WhisperModelState {
    JNIEnv *env = nullptr;
    struct transcribe_session *session = nullptr;

    volatile int cancel_flag = 0;
    jobject partial_result_instance;
    jmethodID partial_result_method;

    size_t ticker = 0;
    size_t last_partial_emit = 0;
    size_t last_exit = 0;
    size_t inference_time = 0;
    size_t preview_every = 0;

    // FUTO's "acft" models are fine-tuned to work with a shortened audio window; stock
    // models (e.g. Whisper large-v3-turbo) must use the full 30 s window or quality drops.
    bool shorten_audio_ctx = false;
    bool using_gpu = false;

    // For falling back from GPU to CPU in the middle of a request
    std::string model_path;
    int n_threads = 4;
    std::string last_error;
    std::string gpu_log;  // engine log lines from the last GPU attempt
};

// Custom build: while a GPU attempt is running, keep the engine's log lines so they can be
// shown in Voice Input settings ("Last voice input run") to diagnose GPU problems.
static std::mutex g_log_mutex;
static std::deque<std::string> g_engine_log;
static std::atomic<bool> g_capture_log{false};

// Crash-safe copy of the GPU attempt log: every line is written and flushed to a file as it
// happens, so if the GPU driver crashes the whole process, the next run can still show it.
static FILE *g_gpu_log_file = nullptr;
static void gpu_log_file_open(const std::string &path) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if(g_gpu_log_file) { fclose(g_gpu_log_file); g_gpu_log_file = nullptr; }
    if(!path.empty()) g_gpu_log_file = fopen(path.c_str(), "w");
}
static void gpu_log_file_close() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if(g_gpu_log_file) { fclose(g_gpu_log_file); g_gpu_log_file = nullptr; }
}
static void gpu_log_write_locked(const std::string &line) {
    if(g_gpu_log_file) { fputs(line.c_str(), g_gpu_log_file); fputc('\n', g_gpu_log_file); fflush(g_gpu_log_file); }
}
// Our own progress markers, so a crash log shows which step was running
static void gpu_step(const char *what) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::string line = std::string("[step] ") + what;
    gpu_log_write_locked(line);
    if(g_capture_log) g_engine_log.push_back(line);
}

static void engine_log_start() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_engine_log.clear();
    g_capture_log = true;
}

static std::string engine_log_take() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_capture_log = false;
    std::string out;
    for(auto &l : g_engine_log) { out += l; out += "\n"; }
    g_engine_log.clear();
    return out;
}

void log_callback(transcribe_log_level level, const char *msg, void *userdata) {
    if(level >= TRANSCRIBE_LOG_LEVEL_ERROR) {
        AKLOGE("[transcribe:%d] %s", level, msg);
    } else {
        AKLOGI("[transcribe:%d] %s", level, msg);
    }

    if(g_capture_log && msg && level >= TRANSCRIBE_LOG_LEVEL_INFO) {
        std::string line = msg;
        while(!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if(line.empty()) return;
        if(line.size() > 240) line = line.substr(0, 240) + "...";
        const char *tag = level >= TRANSCRIBE_LOG_LEVEL_ERROR ? "E " : level >= TRANSCRIBE_LOG_LEVEL_WARN ? "W " : "";
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_engine_log.push_back(std::string(tag) + line);
        gpu_log_write_locked(std::string(tag) + line);
        while(g_engine_log.size() > 80) g_engine_log.pop_front();
    }
}

bool abort_callback(void* userdata) {
    auto *state = reinterpret_cast<WhisperModelState *>(userdata);
    if(state->cancel_flag) {
        return true;
    }

    /*
    if(state->partial_result_instance && state->partial_result_method) {
        const char* text = transcribe_full_text(state->session);
        if(text == nullptr) {
            AKLOGE("No text?");
        } else if(text[0] == '\0') {
            AKLOGE("Text is blank?");
        } else {
            jstring pjstr = string2jstring(state->env, text);
            state->env->CallVoidMethod(state->partial_result_instance, state->partial_result_method,
                                       pjstr);
            state->env->DeleteLocalRef(pjstr);
        }
    }
    */
    return false;
}

static transcribe_status open_session(const std::string &path, transcribe_backend_request backend, int n_threads, struct transcribe_session **out) {
    struct transcribe_model_load_params mlparams;
    transcribe_model_load_params_init(&mlparams);
    mlparams.backend = backend;

    struct transcribe_session_params sparams;
    transcribe_session_params_init(&sparams);
    sparams.n_threads = n_threads;

    return transcribe_open(path.c_str(), &mlparams, &sparams, out);
}

// Keyboard (custom build): openNative(path, useGpu)
// From libvoice_opencl_shim.so (Android builds with the GPU engine); weak so desktop test
// builds without the shim still link.
extern "C" void voice_opencl_cache_stats(int *hits, int *compiles, int *writes, int *bad) __attribute__((weak));

// Compiled GPU programs are cached next to the GPU log (in the app's no-backup files dir), so
// the ~5 s kernel compile happens once instead of on every model load.
static void set_gpu_program_cache_dir(const std::string &gpu_log_path) {
    auto slash = gpu_log_path.find_last_of('/');
    if(slash == std::string::npos) return;
    std::string dir = gpu_log_path.substr(0, slash) + "/opencl_cache";
    mkdir(dir.c_str(), 0700);
    setenv("VOICE_CL_CACHE_DIR", dir.c_str(), 1);
}

static std::string gpu_cache_summary() {
    if(!voice_opencl_cache_stats) return "";
    int hits = 0, compiles = 0, writes = 0, bad = 0;
    voice_opencl_cache_stats(&hits, &compiles, &writes, &bad);
    return "GPU programs: " + std::to_string(hits) + " loaded from cache, " + std::to_string(compiles) +
           " compiled" + (bad ? ", " + std::to_string(bad) + " stale cache entries replaced" : std::string());
}

// Reads general.architecture from a GGUF file header without loading the model ("" if the file
// isn't GGUF, e.g. FUTO's legacy Whisper .bin files).
static std::string read_gguf_architecture(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if(!f) return "";
    auto rd = [&](void *p, size_t n) { return fread(p, 1, n, f) == n; };
    auto skip = [&](uint64_t n) { return fseek(f, (long)n, SEEK_CUR) == 0; };
    std::string result;
    char magic[4]; uint32_t version = 0; uint64_t n_tensors = 0, n_kv = 0;
    if(rd(magic, 4) && memcmp(magic, "GGUF", 4) == 0 && rd(&version, 4) && version >= 2 &&
       rd(&n_tensors, 8) && rd(&n_kv, 8)) {
        // value sizes for GGUF scalar types 0..12 (8 = string, 9 = array are variable)
        static const int sizes[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        std::function<bool(uint32_t)> skip_value = [&](uint32_t type) -> bool {
            if(type == 8) { uint64_t len; return rd(&len, 8) && skip(len); }
            if(type == 9) {
                uint32_t item; uint64_t count;
                if(!rd(&item, 4) || !rd(&count, 8)) return false;
                if(item < 13 && sizes[item] > 0) return skip(count * (uint64_t)sizes[item]);
                for(uint64_t i = 0; i < count; i++) if(!skip_value(item)) return false;
                return true;
            }
            return type < 13 && sizes[type] > 0 && skip((uint64_t)sizes[type]);
        };
        for(uint64_t i = 0; i < n_kv && i < 512; i++) {
            uint64_t klen; if(!rd(&klen, 8) || klen > 4096) break;
            std::string key(klen, '\0'); if(!rd(&key[0], klen)) break;
            uint32_t type; if(!rd(&type, 4)) break;
            if(key == "general.architecture" && type == 8) {
                uint64_t len; if(!rd(&len, 8) || len > 4096) break;
                result.resize(len); if(!rd(&result[0], len)) result.clear();
                break;
            }
            if(!skip_value(type)) break;
        }
    }
    fclose(f);
    return result;
}

// Loads ggml's backend modules (CPU, and the optional OpenCL GPU module) once. With
// GGML_BACKEND_DL even the CPU backend is a module, so this must run before any model opens.
// First try the folder libtranscribe.so lives in; if that finds no devices, use the app's
// native library folder passed from Kotlin.
static std::string g_backend_init_log;
static void ensure_backends(const std::string &native_lib_dir) {
    static std::mutex m;
    static bool done = false;
    std::lock_guard<std::mutex> lock(m);
    if(done && transcribe_backend_device_count() > 0) return;

    engine_log_start();
    transcribe_init_backends_default();
    if(transcribe_backend_device_count() == 0 && !native_lib_dir.empty()) {
        transcribe_init_backends(native_lib_dir.c_str());
    }
    g_backend_init_log = engine_log_take();
    done = true;
    AKLOGI("[voice] %d compute device(s) available", transcribe_backend_device_count());
}

// True if the model's main compute device is a GPU (asks the device, not the backend name:
// the name can be empty until the first run).
static bool session_on_gpu(struct transcribe_session *session) {
    if(!session) return false;
    struct transcribe_backend_device dev;
    transcribe_backend_device_init(&dev);
    if(transcribe_model_get_device(transcribe_get_model(session), &dev) != TRANSCRIBE_OK) return false;
    return dev.device_type == TRANSCRIBE_DEVICE_TYPE_GPU || dev.device_type == TRANSCRIBE_DEVICE_TYPE_IGPU;
}

// Keyboard (custom build): openNative(path, useGpu, nativeLibDir)
static jlong WhisperGGML_open(JNIEnv *env, jclass clazz, jstring model_dir, jboolean use_gpu, jstring native_lib_dir, jstring gpu_log_path) {
    std::string model_dir_str = jstring2string(env, model_dir);
    if(use_gpu) {
        std::string log_path = gpu_log_path ? jstring2string(env, gpu_log_path) : "";
        gpu_log_file_open(log_path);
        set_gpu_program_cache_dir(log_path);
        gpu_step("loading backends");
    }
    ensure_backends(native_lib_dir ? jstring2string(env, native_lib_dir) : "");

    auto *state = new WhisperModelState();

    AKLOGI("[voice] WhisperGGML_open from %s (gpu=%d)", model_dir_str.c_str(), (int)use_gpu);

    // Big phone cores: use up to 6 threads (S25 Ultra has 8 cores)
    long num_procs = sysconf(_SC_NPROCESSORS_ONLN) - 2;
    if(num_procs < 2) num_procs = 2;
    if(num_procs > 6) num_procs = 6;

    // Only Parakeet-family models (Parakeet, Nemotron) are confirmed to compute correctly on the
    // S25 Ultra's GPU. Whisper models produced garbage text there ("!....London, oh"), which
    // the empty-output check can't catch, so other model types always run on the CPU. Decided
    // from the file header, before the GPU is touched.
    if(use_gpu) {
        std::string arch = read_gguf_architecture(model_dir_str);
        if(arch != "parakeet") {
            gpu_step(("model type '" + (arch.empty() ? std::string("whisper (.bin)") : arch) + "' not allowed on GPU, using CPU").c_str());
            gpu_log_file_close();
            state->last_error = "GPU not used for " + (arch.empty() ? std::string("Whisper") : arch) +
                    " models (they gave wrong text on this GPU), used CPU";
            use_gpu = JNI_FALSE;
        }
    }

    transcribe_status result = TRANSCRIBE_ERR_GGUF;
    if(use_gpu) engine_log_start();
    if(use_gpu) gpu_step("opening model on GPU");
    if(use_gpu) {
        // Stage 1.3: try the GPU first (AUTO picks the OpenCL GPU device when its module
        // loaded), fall back to the CPU if no GPU is available
        result = open_session(model_dir_str, TRANSCRIBE_BACKEND_AUTO, (int)num_procs, &state->session);
        if(result == TRANSCRIBE_OK && state->session && session_on_gpu(state->session)) {
            state->using_gpu = true;
            gpu_step("model opened on GPU");
            { std::string c = gpu_cache_summary(); if(!c.empty()) gpu_step(c.c_str()); }
        } else if(result == TRANSCRIBE_OK && state->session) {
            // Opened fine but on the CPU: no usable GPU device (e.g. OpenCL driver not available)
            state->gpu_log = g_backend_init_log + engine_log_take();
            state->last_error = "No GPU available (OpenCL driver could not be loaded), used CPU";
        } else {
            AKLOGE("[voice] GPU open failed (%s), falling back to CPU", transcribe_status_string(result));
            state->gpu_log = g_backend_init_log + engine_log_take();
            state->last_error = std::string("GPU could not start (") + transcribe_status_string(result) + "), used CPU";
            if(state->session) { transcribe_session_free(state->session); state->session = nullptr; }
        }
    }

    if(use_gpu && !state->using_gpu) gpu_log_file_close();

    if(!state->session) {
        // (If AUTO already landed on the CPU above, that session is reused.)
        result = open_session(model_dir_str, TRANSCRIBE_BACKEND_CPU, (int)num_procs, &state->session);
    }

    if(!state->session || result != TRANSCRIBE_OK){
        AKLOGE("Failed to open voice model '%s': %s", model_dir_str.c_str(), transcribe_status_string(result));
        delete state;
        return 0L;
    }

    // FUTO's legacy .bin Whisper models are all "acft" fine-tunes; GGUF models only if named so
    bool is_gguf = false;
    if(FILE *f = fopen(model_dir_str.c_str(), "rb")) {
        char magic[4] = {0};
        is_gguf = fread(magic, 1, 4, f) == 4 && memcmp(magic, "GGUF", 4) == 0;
        fclose(f);
    }
    state->shorten_audio_ctx = !is_gguf || model_dir_str.find("acft") != std::string::npos;
    state->model_path = model_dir_str;
    state->n_threads = (int)num_procs;

    transcribe_set_abort_callback(state->session, abort_callback, (void*)state);
    return reinterpret_cast<jlong>(state);
}

static std::string narrowLanguageCode(const char *language, struct transcribe_capabilities *capabilities) {
    size_t len_ref = strlen(language);

    std::string result;
    bool exact_match = false;
    for(int i=0; i<capabilities->n_languages; i++) {
        size_t len_cmp = strlen(capabilities->languages[i]);
        if(strncmp(capabilities->languages[i], language, std::min(len_ref, len_cmp)) == 0) {
            if(len_cmp == len_ref) return capabilities->languages[i];

            if(result.empty()) result = capabilities->languages[i];
        }
    }

    return result;
}

static std::string languageArrayToMergedString(JNIEnv *env, jobjectArray languages, struct transcribe_capabilities *capabilities) {
    std::string language_hint;
    int num_languages = env->GetArrayLength(languages);
    for (int i=0; i<num_languages; i++) {
        jstring jstr = static_cast<jstring>(env->GetObjectArrayElement(languages, i));
        std::string str = jstring2string(env, jstr);

        std::string narrowed = narrowLanguageCode(str.c_str(), capabilities);
        if(narrowed.empty() && str == "no") {
            // we refer to norwegian as "no" but parakeet refers to it by "nb" (Norwegian Bokmål)
            narrowed = narrowLanguageCode("nb", capabilities);
        }

        language_hint.append(narrowed);
        if(i < (num_languages - 1) && !narrowed.empty()) {
            language_hint.append(",");
        }
    }

    return language_hint;
}

// Keyboard (custom build) signature:
// Output that can't be real speech: a character repeated 6+ times in a row ("êêêêêê"), the same
// word 5+ times in a row, or more letters than anyone can say in that time (~25 per second).
// Seen from the GPU on very short clips; the CPU transcribes the same audio correctly.
static bool looks_garbled(const char *text, int num_samples) {
    if(!text) return false;
    int run = 1, letters = 0;
    unsigned char prev = 0;
    for(const unsigned char *c = (const unsigned char *)text; *c; c++) {
        if(isspace(*c)) { prev = 0; run = 1; continue; }
        if((*c & 0xC0) != 0x80) letters++;           // count UTF-8 characters, not bytes
        if((*c & 0xC0) == 0x80) continue;            // compare characters by their first byte
        if(*c == prev && *c != '.' && !isdigit(*c)) { if(++run >= 6) return true; }
        else run = 1;
        prev = *c;
    }
    const double seconds = std::max(0.5, num_samples / 16000.0);
    if(letters > 25.0 * seconds + 10) return true;
    // Same word repeated
    std::string t(text), last, word;
    int repeats = 1;
    for(size_t i = 0; i <= t.size(); i++) {
        if(i == t.size() || isspace((unsigned char)t[i])) {
            if(!word.empty()) {
                if(word == last) { if(++repeats >= 5) return true; } else repeats = 1;
                last = word; word.clear();
            }
        } else word += t[i];
    }
    return false;
}

// inferNative(handle, samples, glossary, textContext, languages, bailLanguages, decodingMode, suppressNonSpeech)
static jstring WhisperGGML_infer(JNIEnv *env, jobject instance, jlong handle, jfloatArray samples_array, jstring prompt, jstring text_context, jobjectArray languages, jobjectArray bail_languages, jint decoding_mode, jboolean suppress_non_speech) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    state->cancel_flag = 0;

    struct transcribe_capabilities capabilities;
    transcribe_capabilities_init(&capabilities);
    transcribe_model_get_capabilities(transcribe_get_model(state->session), &capabilities);

    std::string language_hint = languageArrayToMergedString(env, languages, &capabilities);
    std::string bail_language_hint = languageArrayToMergedString(env, bail_languages, &capabilities);
    std::string prompt_str;

    if(strcmp(transcribe_model_arch_string(transcribe_get_model(state->session)),
            "parakeet") == 0) {
        if(language_hint.find(',') != std::string::npos) {
            AKLOGE("Parakeet/Nemotron do not support multilingual, clearing hint");
            language_hint = "";
        }
    }

    state->env = env;
    state->partial_result_instance = instance;
    state->partial_result_method = env->GetMethodID(
        env->GetObjectClass(instance),
        "invokePartialResult",
          "(Ljava/lang/String;)V");

    int num_samples = env->GetArrayLength(samples_array);
    jfloat *samples = env->GetFloatArrayElements(samples_array, nullptr);


    struct transcribe_whisper_run_ext wp;

    struct transcribe_run_params params;
    transcribe_run_params_init(&params);
    params.task = TRANSCRIBE_TASK_TRANSCRIBE;
    params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
    params.language = language_hint.c_str();
    params.keep_special_tags = false;
    //params.spec_k_drafts = 0;
    params.bail_languages = bail_language_hint.c_str();

    if(transcribe_model_accepts_ext_kind(
            transcribe_get_model(state->session),
            TRANSCRIBE_EXT_SLOT_RUN,
            TRANSCRIBE_EXT_KIND_WHISPER_RUN
    )) {
        transcribe_whisper_run_ext_init(&wp);

        if(state->shorten_audio_ctx) {
            wp.audio_ctx = std::max(160, std::min(1500, (int)ceil((double)num_samples / (double)(320.0)) + 32));
        }

        // The keyboard passes an already formatted glossary, e.g. "(Glossary: FUTO, Zainab)"
        if(prompt) {
            prompt_str = jstring2string(env, prompt);
        }

        // Stage 1.1: the text already in the text box, so names, casing, punctuation style
        // and topic carry over into the transcription
        if(text_context) {
            auto ctx = jstring2string(env, text_context);
            if(!ctx.empty()) {
                if(!prompt_str.empty()) prompt_str.append(" ");
                prompt_str.append(ctx);
            }
        }

        if(language_hint == "zh") {
            prompt_str.append(" [简体]");
        }

        wp.initial_prompt = prompt_str.c_str();
        AKLOGI("Initial prompt for [%s]: %s", language_hint.c_str(), prompt_str.c_str());

        params.family = &wp.ext;
    }

    if(state->using_gpu) gpu_step("running speech model on GPU (first run compiles GPU kernels)");
    // Very short clips (under ~2 s) sometimes came back as garbage on the phone's GPU while the
    // CPU handled them fine: give the GPU at least 2 s, padded with silence (checked on the CPU
    // that the padding doesn't change the words).
    const int min_gpu_samples = 2 * 16000;
    std::vector<float> padded;
    const float *run_samples = samples;
    int run_count = num_samples;
    if(state->using_gpu && num_samples < min_gpu_samples) {
        padded.assign(samples, samples + num_samples);
        padded.resize(min_gpu_samples, 0.0f);
        run_samples = padded.data();
        run_count = min_gpu_samples;
    }
    auto run_start = std::chrono::steady_clock::now();
    auto result = transcribe_run(state->session, run_samples, run_count, &params);
    const long gpu_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - run_start).count();

    const bool attempted_gpu = state->using_gpu;
    // GPU safety net: if the GPU run failed, or produced no text for over a second of audio,
    // re-open the model on the CPU and run again, so the user still gets their text.
    if(state->using_gpu && result != TRANSCRIBE_ERR_ABORTED && result != TRANSCRIBE_ERR_LANG_BAILED) {
        bool empty = true;
        if(result == TRANSCRIBE_OK) {
            const char *t = transcribe_full_text(state->session);
            if(t) for(const char *c = t; *c; c++) if(!isspace((unsigned char)*c)) { empty = false; break; }
        }
        if(result != TRANSCRIBE_OK || (empty && num_samples > 16000)) {
            state->last_error = result != TRANSCRIBE_OK
                    ? std::string("GPU run failed (") + transcribe_status_string(result) + "), used CPU instead"
                    : std::string("GPU returned no text, used CPU instead");
            state->last_error += " (GPU attempt took " + std::to_string(gpu_ms) + " ms)";
            AKLOGE("[voice] %s", state->last_error.c_str());

            state->gpu_log = g_backend_init_log + engine_log_take();
            transcribe_session_free(state->session);
            state->session = nullptr;
            state->using_gpu = false;
            auto r2 = open_session(state->model_path, TRANSCRIBE_BACKEND_CPU, state->n_threads, &state->session);
            if(r2 == TRANSCRIBE_OK && state->session) {
                transcribe_set_abort_callback(state->session, abort_callback, (void*)state);
                result = transcribe_run(state->session, samples, num_samples, &params);
            } else {
                state->last_error += std::string("; CPU re-open failed: ") + transcribe_status_string(r2);
                env->ReleaseFloatArrayElements(samples_array, samples, JNI_ABORT);
                return 0;
            }
        }
    }
    // Garbled GPU text (e.g. "êêêêêê"): run just this clip on the CPU. The GPU session is kept,
    // so the next clip still uses the GPU.
    std::string cpu_redo_text;
    bool cpu_redo = false;
    if(state->using_gpu && result == TRANSCRIBE_OK && looks_garbled(transcribe_full_text(state->session), num_samples)) {
        AKLOGE("[voice] GPU output looks garbled, re-running this clip on the CPU");
        struct transcribe_session *cpu = nullptr;
        auto r2 = open_session(state->model_path, TRANSCRIBE_BACKEND_CPU, state->n_threads, &cpu);
        if(r2 == TRANSCRIBE_OK && cpu) {
            transcribe_set_abort_callback(cpu, abort_callback, (void*)state);
            auto r3 = transcribe_run(cpu, samples, num_samples, &params);
            if(r3 == TRANSCRIBE_OK) {
                const char *t = transcribe_full_text(cpu);
                cpu_redo_text = t ? t : "";
                cpu_redo = true;
                state->last_error = "GPU gave garbled text for this clip (" + std::to_string(num_samples / 16) +
                        " ms of audio), used CPU for it";
            }
            transcribe_session_free(cpu);
        }
    }
    env->ReleaseFloatArrayElements(samples_array, samples, JNI_ABORT);
    if(attempted_gpu) gpu_step("GPU attempt finished");
    if(attempted_gpu && state->gpu_log.empty()) state->gpu_log = g_backend_init_log + engine_log_take();
    if(attempted_gpu) gpu_log_file_close();

    std::string output;
    if(result == TRANSCRIBE_ERR_LANG_BAILED) {
        auto language = transcribe_detected_language(state->session);
        if(language == nullptr || language[0] == '\0') {
            AKLOGE("Bailed due to language detection which is null/blank!");
            return 0;
        }

        output = "<>CANCELLED<> lang=";
        output.append(language);

        jstring jstr = string2jstring(env, output.c_str());
        return jstr;
    }

    if(result == TRANSCRIBE_ERR_ABORTED) {
        output = "<>CANCELLED<> flag";

        jstring jstr = string2jstring(env, output.c_str());
        return jstr;
    }

    if(result != TRANSCRIBE_OK) {
        AKLOGE("Failed to run! Status: %s", transcribe_status_string(result));
        state->last_error = std::string("Run failed: ") + transcribe_status_string(result);
        return 0;
    }

    if(cpu_redo) return string2jstring(env, cpu_redo_text.c_str());
    auto txt = transcribe_full_text(state->session);
    return string2jstring(env, txt);
}

static jstring WhisperGGML_lastError(JNIEnv *env, jclass clazz, jlong handle) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    return env->NewStringUTF(state ? state->last_error.c_str() : "");
}

static jstring WhisperGGML_gpuLog(JNIEnv *env, jclass clazz, jlong handle) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    if(!state) return env->NewStringUTF("");
    std::string log = state->gpu_log;
    state->gpu_log.clear();
    return string2jstring(env, log.c_str());
}

static jboolean WhisperGGML_startStream(JNIEnv *env, jobject instance, jlong handle, jstring prompt, jobjectArray languages, jobjectArray bail_languages, jint tier) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    state->cancel_flag = 0;

    struct transcribe_capabilities capabilities;
    transcribe_capabilities_init(&capabilities);
    transcribe_model_get_capabilities(transcribe_get_model(state->session), &capabilities);

    std::string language_hint = languageArrayToMergedString(env, languages, &capabilities);
    std::string bail_language_hint = languageArrayToMergedString(env, bail_languages, &capabilities);
    std::string prompt_str = prompt ? jstring2string(env, prompt) : "";

    if(strcmp(transcribe_model_arch_string(transcribe_get_model(state->session)),
              "parakeet") == 0) {
        if(language_hint.find(',') != std::string::npos) {
            AKLOGE("Parakeet/Nemotron do not support multilingual, clearing hint");
            language_hint = "";
        }
    }

    state->env = env;
    state->partial_result_instance = instance;
    state->partial_result_method = env->GetMethodID(
            env->GetObjectClass(instance),
            "invokePartialResult",
            "(Ljava/lang/String;)V");

    struct transcribe_asr4all_stream_ext a4ap;
    struct transcribe_parakeet_stream_ext pkp;

    struct transcribe_stream_params s_params;
    struct transcribe_run_params r_params;
    transcribe_stream_params_init(&s_params);
    transcribe_run_params_init(&r_params);

    r_params.task = TRANSCRIBE_TASK_TRANSCRIBE;
    r_params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
    r_params.language = language_hint.c_str();
    r_params.keep_special_tags = false;
    r_params.bail_languages = bail_language_hint.c_str();

    if(transcribe_model_accepts_ext_kind(
            transcribe_get_model(state->session),
            TRANSCRIBE_EXT_SLOT_STREAM,
            TRANSCRIBE_EXT_KIND_ASR4ALL_STREAM
    )) {
        transcribe_asr4all_stream_ext_init(&a4ap);

        a4ap.custom_dictionary = prompt_str.empty() ? NULL : prompt_str.c_str();

        if(tier <= 0) {
            a4ap.commit_frames = 16;
            a4ap.right_ctx_frames = 4;
            state->preview_every = 2000;
        } else if(tier == 1) {
            a4ap.commit_frames = 32;
            a4ap.right_ctx_frames = 8;
            state->preview_every = 4000;
        } else if(tier >= 2) {
            a4ap.commit_frames = 64;
            a4ap.right_ctx_frames = 16;
            state->preview_every = 8000;
        } /*else {
            a4ap.commit_frames = 512;
            a4ap.right_ctx_frames = 32;
            a4ap.lookahead_frames = 162;

            if(tier == 1) a4ap.commit_frames = 128;
            if(tier == 2) a4ap.commit_frames = 256;
        }*/

        s_params.family = &a4ap.ext;
    }else if(transcribe_model_accepts_ext_kind(
            transcribe_get_model(state->session),
            TRANSCRIBE_EXT_SLOT_STREAM,
            TRANSCRIBE_EXT_KIND_PARAKEET_STREAM
    )) {
        transcribe_parakeet_stream_ext_init(&pkp);

        // att_context_right = 0 doesn't really run on mobile
        if(tier <= 0) {
            if(strcmp(transcribe_model_variant_string(transcribe_get_model(state->session)), "nemotron-3.5-asr-streaming-0.6b") == 0) {
                pkp.att_context_right = 3;
            } else {
                pkp.att_context_right = 1;
            }
        } else if(tier == 1) {
            pkp.att_context_right = 6;
        } else {
            pkp.att_context_right = 13;
        }

        state->preview_every = 0;

        s_params.family = &pkp.ext;
    }

    //AKLOGE("variant is %s", transcribe_model_variant_string(transcribe_get_model(state->session)));

    state->ticker = 0;
    state->inference_time = 0;

    auto result = transcribe_stream_begin(state->session, &r_params, &s_params) == TRANSCRIBE_OK;
    if(!result && a4ap.custom_dictionary != NULL && s_params.family == &a4ap.ext) {
        // needs updated gguf
        AKLOGE("Failed to start stream with custom dictionary. Retrying with it cleared.");
        a4ap.custom_dictionary = NULL;
        result = transcribe_stream_begin(state->session, &r_params, &s_params) == TRANSCRIBE_OK;
        if(!result) {
            AKLOGE("Failed anyway.");
        }
    }

    return result;
}

static jboolean WhisperGGML_feedStream(JNIEnv *env, jobject instance, jlong handle, jfloatArray samples_array) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    state->cancel_flag = 0;

    state->env = env;
    state->partial_result_instance = instance;
    state->partial_result_method = env->GetMethodID(
            env->GetObjectClass(instance),
            "invokePartialResult",
            "(Ljava/lang/String;)V");

    int num_samples = env->GetArrayLength(samples_array);
    jfloat *samples = env->GetFloatArrayElements(samples_array, nullptr);

    auto t0 = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    bool is_ahead = (state->last_exit - t0) > ((num_samples / 16) * 3 / 2);

    transcribe_status status;
    struct transcribe_stream_update result;
    transcribe_stream_update_init(&result);
    if((status = transcribe_stream_feed(state->session, samples, num_samples, &result)) != TRANSCRIBE_OK) {
        AKLOGE("stream feed failed! %s", transcribe_status_string(status));
        return false;
    }

    auto t1 = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();


    bool preview = (state->preview_every > 0) && (state->ticker - state->last_partial_emit) > state->preview_every;
    if(is_ahead && preview && (status = transcribe_stream_feed(state->session, nullptr, 0, &result)) != TRANSCRIBE_OK) {
        AKLOGE("stream tail feed failed! %s", transcribe_status_string(status));
        // non critical
    }

    if(preview) state->last_partial_emit = state->ticker;
    state->ticker += num_samples;

    auto t2 = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    struct transcribe_stream_text text_result;
    transcribe_stream_text_init(&text_result);
    if ((status = transcribe_stream_get_text(state->session, &text_result)) != TRANSCRIBE_OK) {
        AKLOGE("stream get_text failed! %s", transcribe_status_string(status));
        return false;
    }


    auto t3 = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    //AKLOGI("[%d] processed %d samples in %d ms + %d ms + %d ms (%s)", state->ticker, num_samples, (int)(t1-t0), (int)(t2-t1), (int)(t3-t1), preview ? "previewed" : "no");

    if (text_result.full_text != nullptr) {
        jstring pjstr = string2jstring(env, text_result.full_text);
        state->env->CallVoidMethod(state->partial_result_instance, state->partial_result_method,
                                   pjstr);
        state->env->DeleteLocalRef(pjstr);
    }

    state->last_exit = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    state->inference_time += state->last_exit - t0;

    double xRT = (((double)state->ticker) / 16.0) / ((double)(state->inference_time));
    //AKLOGI("xRT = %.2f txt=[%s]", (float)xRT, text_result.full_text);
    if(xRT < 0.8 && (state->ticker > 32000)) {
        AKLOGE("Falling back to offline.");
        std::string s = text_result.full_text;
        s.append(" [...]");
        jstring pjstr = string2jstring(env, s.c_str());
        state->env->CallVoidMethod(state->partial_result_instance, state->partial_result_method, pjstr);
        state->env->DeleteLocalRef(pjstr);
        return false;
    }

    return true;
}

static jstring WhisperGGML_stopStream(JNIEnv *env, jobject instance, jlong handle, jboolean finalize) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    state->cancel_flag = 0;
    state->env = env;

    if(!finalize) {
        transcribe_stream_reset(state->session);
        return nullptr;
    }

    transcribe_status status;
    struct transcribe_stream_update result;
    transcribe_stream_update_init(&result);
    if((status = transcribe_stream_finalize(state->session, &result)) != TRANSCRIBE_OK) {
        AKLOGE("stream finalize failed! %s", transcribe_status_string(status));
        return nullptr;
    }

    struct transcribe_stream_text text_result;
    transcribe_stream_text_init(&text_result);
    if((status = transcribe_stream_get_text(state->session, &text_result)) != TRANSCRIBE_OK) {
        AKLOGE("stream get_text failed! %s", transcribe_status_string(status));
        return nullptr;
    }

    jstring pjstr = string2jstring(state->env, text_result.full_text);
    return pjstr;
}


static void WhisperGGML_close(JNIEnv *env, jclass clazz, jlong handle) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    if(!state) return;

    transcribe_session_free(state->session);

    delete state;
}

static jboolean WhisperGGML_isUsingGpu(JNIEnv *env, jclass clazz, jlong handle) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    return (state && state->using_gpu) ? JNI_TRUE : JNI_FALSE;
}

static void WhisperGGML_cancel(JNIEnv *env, jclass clazz, jlong handle) {
    auto *state = reinterpret_cast<WhisperModelState *>(handle);
    state->cancel_flag = 1;
}

static const JNINativeMethod sMethods[] = {
        {
                const_cast<char *>("openNative"),
                const_cast<char *>("(Ljava/lang/String;ZLjava/lang/String;Ljava/lang/String;)J"),
                reinterpret_cast<void *>(WhisperGGML_open)
        },
        {
                const_cast<char *>("inferNative"),
                const_cast<char *>("(J[FLjava/lang/String;Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;IZ)Ljava/lang/String;"),
                reinterpret_cast<void *>(WhisperGGML_infer)
        },
        {
                const_cast<char *>("isUsingGpuNative"),
                const_cast<char *>("(J)Z"),
                reinterpret_cast<void *>(WhisperGGML_isUsingGpu)
        },
        {
                const_cast<char *>("gpuLogNative"),
                const_cast<char *>("(J)Ljava/lang/String;"),
                reinterpret_cast<void *>(WhisperGGML_gpuLog)
        },
        {
                const_cast<char *>("lastErrorNative"),
                const_cast<char *>("(J)Ljava/lang/String;"),
                reinterpret_cast<void *>(WhisperGGML_lastError)
        },
        {
                const_cast<char *>("cancelNative"),
                const_cast<char *>("(J)V"),
                reinterpret_cast<void *>(WhisperGGML_cancel)
        },
        {
                const_cast<char *>("closeNative"),
                const_cast<char *>("(J)V"),
                reinterpret_cast<void *>(WhisperGGML_close)
        }
};

int register_WhisperGGML(JNIEnv *env) {
    transcribe_log_set(log_callback, nullptr);

    const char *const kClassPathName = "org/futo/voiceinput/shared/ggml/WhisperGGML";
    return registerNativeMethods(env, kClassPathName, sMethods, NELEMS(sMethods));
}