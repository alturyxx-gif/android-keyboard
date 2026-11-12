/*
 * Custom build: OpenCL shim for the GPU voice engine (ggml-opencl).
 *
 * Apps can't ship the GPU vendor's OpenCL driver, and linking straight to the phone's
 * libOpenCL.so would make the whole voice engine fail to load on a phone that doesn't expose
 * it. Instead ggml-opencl links to this small library (libvoice_opencl_shim.so). On first use
 * it dlopen()s the phone's real driver (declared via <uses-native-library> in the manifest)
 * and forwards every call to it. If no driver can be loaded, clGetPlatformIDs reports
 * "no platforms", ggml sees no GPU, and voice input simply runs on the CPU.
 *
 * Generated from the prototypes in Khronos OpenCL-Headers (CL/cl.h) for the functions
 * ggml-opencl uses.
 */
#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <dlfcn.h>
#include <stddef.h>
#include <pthread.h>

#ifndef CL_PLATFORM_NOT_FOUND_KHR
#define CL_PLATFORM_NOT_FOUND_KHR -1001
#endif

static void *g_driver = NULL;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void shim_load(void) {
    static const char *candidates[] = {
        "libOpenCL.so",                    /* resolved via <uses-native-library> (Android 12+) */
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL_adreno.so",
        NULL,
    };
    for (int i = 0; candidates[i] && !g_driver; i++) {
        g_driver = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
    }
}

static void *shim_sym(const char *name) {
    pthread_once(&g_once, shim_load);
    return g_driver ? dlsym(g_driver, name) : NULL;
}

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id * device_list, const char * options, void (CL_CALLBACK * pfn_notify)(cl_program program, void * user_data), void * user_data) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_program program, cl_uint num_devices, const cl_device_id * device_list, const char * options, void (CL_CALLBACK * pfn_notify)(cl_program program, void * user_data), void * user_data);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clBuildProgram"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(program, num_devices, device_list, options, pfn_notify, user_data);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void * host_ptr, cl_int * errcode_ret) {
    typedef cl_mem (CL_API_CALL *fn_t)(cl_context context, cl_mem_flags flags, size_t size, void * host_ptr, cl_int * errcode_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clCreateBuffer"); tried = 1; }
    if (!fn) { if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; return NULL; }
    return fn(context, flags, size, host_ptr, errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBufferWithProperties(cl_context context, const cl_mem_properties * properties, cl_mem_flags flags, size_t size, void * host_ptr, cl_int * errcode_ret) {
    typedef cl_mem (CL_API_CALL *fn_t)(cl_context context, const cl_mem_properties * properties, cl_mem_flags flags, size_t size, void * host_ptr, cl_int * errcode_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clCreateBufferWithProperties"); tried = 1; }
    if (!fn) { if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; return NULL; }
    return fn(context, properties, flags, size, host_ptr, errcode_ret);
}

CL_API_ENTRY CL_API_PREFIX__VERSION_1_2_DEPRECATED cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int * errcode_ret) {
    typedef CL_API_PREFIX__VERSION_1_2_DEPRECATED cl_command_queue (CL_API_CALL *fn_t)(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int * errcode_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clCreateCommandQueue"); tried = 1; }
    if (!fn) { if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; return NULL; }
    return fn(context, device, properties, errcode_ret);
}

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(const cl_context_properties * properties, cl_uint num_devices, const cl_device_id * devices, void (CL_CALLBACK * pfn_notify)(const char * errinfo, const void * private_info, size_t cb, void * user_data), void * user_data, cl_int * errcode_ret) {
    typedef cl_context (CL_API_CALL *fn_t)(const cl_context_properties * properties, cl_uint num_devices, const cl_device_id * devices, void (CL_CALLBACK * pfn_notify)(const char * errinfo, const void * private_info, size_t cb, void * user_data), void * user_data, cl_int * errcode_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clCreateContext"); tried = 1; }
    if (!fn) { if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; return NULL; }
    return fn(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format * image_format, const cl_image_desc * image_desc, void * host_ptr, cl_int * errcode_ret) {
    typedef cl_mem (CL_API_CALL *fn_t)(cl_context context, cl_mem_flags flags, const cl_image_format * image_format, const cl_image_desc * image_desc, void * host_ptr, cl_int * errcode_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clCreateImage"); tried = 1; }
    if (!fn) { if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; return NULL; }
    return fn(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program program, const char * kernel_name, cl_int * errcode_ret) {
    typedef cl_kernel (CL_API_CALL *fn_t)(cl_program program, const char * kernel_name, cl_int * errcode_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clCreateKernel"); tried = 1; }
    if (!fn) { if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; return NULL; }
    return fn(program, kernel_name, errcode_ret);
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(cl_context context, cl_uint count, const char ** strings, const size_t * lengths, cl_int * errcode_ret) {
    typedef cl_program (CL_API_CALL *fn_t)(cl_context context, cl_uint count, const char ** strings, const size_t * lengths, cl_int * errcode_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clCreateProgramWithSource"); tried = 1; }
    if (!fn) { if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; return NULL; }
    return fn(context, count, strings, lengths, errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void * buffer_create_info, cl_int * errcode_ret) {
    typedef cl_mem (CL_API_CALL *fn_t)(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void * buffer_create_info, cl_int * errcode_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clCreateSubBuffer"); tried = 1; }
    if (!fn) { if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; return NULL; }
    return fn(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrierWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clEnqueueBarrierWithWaitList"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue, num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clEnqueueCopyBuffer"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void * pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue, cl_mem buffer, const void * pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clEnqueueFillBuffer"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarkerWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clEnqueueMarkerWithWaitList"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue, num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t * global_work_offset, const size_t * global_work_size, const size_t * local_work_size, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t * global_work_offset, const size_t * global_work_size, const size_t * local_work_size, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clEnqueueNDRangeKernel"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clEnqueueReadBuffer"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clEnqueueWriteBuffer"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue command_queue) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clFinish"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL clFlush(cl_command_queue command_queue) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_command_queue command_queue);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clFlush"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id * devices, cl_uint * num_devices) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id * devices, cl_uint * num_devices);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetDeviceIDs"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(platform, device_type, num_entries, devices, num_devices);
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_device_id device, cl_device_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetDeviceInfo"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(device, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_event event, cl_profiling_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetEventProfilingInfo"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(event, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetKernelInfo"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(kernel, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelSubGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name, size_t input_value_size, const void* input_value, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name, size_t input_value_size, const void* input_value, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetKernelSubGroupInfo"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(kernel, device, param_name, input_value_size, input_value, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetKernelWorkGroupInfo"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetPlatformIDs"); tried = 1; }
    if (!fn) { if (num_platforms) *num_platforms = 0; return CL_PLATFORM_NOT_FOUND_KHR; }
    return fn(num_entries, platforms, num_platforms);
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetPlatformInfo"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clGetProgramBuildInfo"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent(cl_event event) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_event event);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clReleaseEvent"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(event);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_mem memobj);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clReleaseMemObject"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(memobj);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program program) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_program program);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clReleaseProgram"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(program);
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void * arg_value) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void * arg_value);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clSetKernelArg"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(kernel, arg_index, arg_size, arg_value);
}

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents(cl_uint num_events, const cl_event * event_list) {
    typedef cl_int (CL_API_CALL *fn_t)(cl_uint num_events, const cl_event * event_list);
    static fn_t fn = NULL; static int tried = 0;
    if (!tried) { fn = (fn_t)shim_sym("clWaitForEvents"); tried = 1; }
    if (!fn) { return CL_INVALID_PLATFORM; }
    return fn(num_events, event_list);
}
