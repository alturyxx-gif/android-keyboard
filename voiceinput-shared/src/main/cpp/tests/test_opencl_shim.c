/*
 * Host test for opencl_shim.c (custom build). Runs against any desktop OpenCL implementation
 * (e.g. PoCL: apt install pocl-opencl-icd). Uses the shim exactly like ggml-opencl does:
 * clCreateProgramWithSource -> clBuildProgram -> clCreateKernel -> run -> clReleaseProgram.
 *
 *   cc -I../OpenCL-Headers ../opencl_shim.c test_opencl_shim.c -ldl -lpthread -o t && ./t
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

void voice_opencl_cache_stats(int *hits, int *compiles, int *writes, int *bad);

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } else { printf("  ok: " __VA_ARGS__); printf("\n"); } } while (0)

static const char *SRC = "kernel void add_k(global float *a, float k) { a[get_global_id(0)] += k; }";

static cl_context ctx; static cl_device_id dev; static cl_command_queue q;

/* Builds the program like ggml, runs it on 64 floats, returns 1 if results are correct. */
static int build_and_run(const char *opts, float k) {
    cl_int err; size_t len = strlen(SRC); const char *s = SRC;
    cl_program p = clCreateProgramWithSource(ctx, 1, &s, &len, &err);
    if (err != CL_SUCCESS) return 0;
    if (clBuildProgram(p, 0, NULL, opts, NULL, NULL) != CL_SUCCESS) {
        char log[4096] = {0}; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        printf("build log: %s\n", log); return 0;
    }
    cl_kernel kern = clCreateKernel(p, "add_k", &err);
    if (err != CL_SUCCESS) return 0;
    float data[64]; for (int i = 0; i < 64; i++) data[i] = (float)i;
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data), data, &err);
    clSetKernelArg(kern, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(kern, 1, sizeof(float), &k);
    size_t gws = 64;
    clEnqueueNDRangeKernel(q, kern, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(q, buf, CL_TRUE, 0, sizeof(data), data, 0, NULL, NULL);
    clFinish(q);
    int ok = 1; for (int i = 0; i < 64; i++) if (data[i] != (float)i + k) ok = 0;
    clReleaseMemObject(buf);
    clReleaseProgram(p);
    return ok;
}

static int count_files(const char *dir) {
    int n = 0; DIR *d = opendir(dir); struct dirent *e;
    if (!d) return 0;
    while ((e = readdir(d))) if (strstr(e->d_name, ".clbin")) n++;
    closedir(d); return n;
}

static void stats(int *h, int *c, int *w, int *b) { voice_opencl_cache_stats(h, c, w, b); }

int main(void) {
    char dir[] = "/tmp/voice_cl_cache_XXXXXX";
    if (!mkdtemp(dir)) return 2;
    setenv("VOICE_CL_CACHE_DIR", dir, 1);

    cl_platform_id plat; cl_uint np = 0;
    if (clGetPlatformIDs(1, &plat, &np) != CL_SUCCESS || np == 0) { printf("no OpenCL platform\n"); return 2; }
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL) != CL_SUCCESS) return 2;
    cl_int err;
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    q = clCreateCommandQueue(ctx, dev, 0, &err);
    int h, c, w, b;

    printf("== First load: compile from source and save\n");
    CHECK(build_and_run("-cl-fast-relaxed-math", 1.0f), "kernel computes correctly");
    stats(&h, &c, &w, &b);
    CHECK(c == 1 && h == 0 && w == 1, "compiled once and saved (compiles=%d hits=%d writes=%d)", c, h, w);
    CHECK(count_files(dir) == 1, "one cache file on disk");

    printf("== Second load: from the cache\n");
    CHECK(build_and_run("-cl-fast-relaxed-math", 2.0f), "cached binary computes correctly");
    stats(&h, &c, &w, &b);
    CHECK(h == 1 && c == 1, "loaded from cache, no recompile (compiles=%d hits=%d)", c, h);

    printf("== Different build options: separate cache entry\n");
    CHECK(build_and_run("-cl-mad-enable", 3.0f), "kernel computes correctly");
    stats(&h, &c, &w, &b);
    CHECK(c == 2 && count_files(dir) == 2, "compiled again for new options (compiles=%d, files=%d)", c, count_files(dir));

    printf("== Corrupted cache file: detected, recompiled, replaced\n");
    DIR *d = opendir(dir); struct dirent *e; char victim[1200] = {0};
    while ((e = readdir(d))) if (strstr(e->d_name, ".clbin")) { snprintf(victim, sizeof(victim), "%s/%s", dir, e->d_name); break; }
    closedir(d);
    FILE *f = fopen(victim, "wb"); fputs("garbage, not a GPU binary", f); fclose(f);
    int ok1 = build_and_run("-cl-fast-relaxed-math", 4.0f);
    int ok2 = build_and_run("-cl-mad-enable", 5.0f);
    CHECK(ok1 && ok2, "both programs still compute correctly");
    stats(&h, &c, &w, &b);
    CHECK(b == 1 && c == 3, "bad binary detected and recompiled (bad=%d compiles=%d hits=%d)", b, c, h);
    CHECK(build_and_run("-cl-fast-relaxed-math", 6.0f), "replacement cache file works");
    int h2; stats(&h2, &c, &w, &b);
    CHECK(h2 > h && c == 3, "and is loaded from cache again (hits=%d compiles=%d)", h2, c);

    printf("== No cache dir set: still works, nothing written\n");
    unsetenv("VOICE_CL_CACHE_DIR");
    int before = count_files(dir);
    CHECK(build_and_run("-cl-fast-relaxed-math", 7.0f), "kernel computes correctly");
    CHECK(count_files(dir) == before, "no files written");

    printf("\n%s (%d failures)\n", failures ? "SOME FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
