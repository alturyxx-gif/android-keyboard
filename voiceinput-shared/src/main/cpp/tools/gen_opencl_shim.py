#!/usr/bin/env python3
"""Generates opencl_shim.c (custom build).

Usage: gen_opencl_shim.py <OpenCL-Headers dir> <ggml-opencl.cpp> <output .c>

Forwards every OpenCL function ggml-opencl uses to the phone's real driver (loaded with
dlopen). The program-related functions are hand-written: they add a compiled-program cache
so the GPU kernels are compiled only once instead of on every model load.
"""
import re, sys

hdr_dir, ggml_src, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(hdr_dir + '/CL/cl.h').read() + open(hdr_dir + '/CL/cl_ext.h').read()
api = set(re.findall(r'CL_API_CALL\s+(cl[A-Z]\w+)\s*\(', src))
used = sorted(set(re.findall(r'\b(cl[A-Z][A-Za-z0-9]+)\b', open(ggml_src).read())) & api)

HAND_WRITTEN = {'clCreateProgramWithSource', 'clBuildProgram', 'clGetProgramBuildInfo',
                'clCreateKernel', 'clReleaseProgram'}
EXTRA_DRIVER = ['clCreateProgramWithBinary', 'clGetProgramInfo', 'clGetDeviceInfo',
                'clGetContextInfo']

def split_params(p):
    parts, depth, cur = [], 0, ''
    for ch in p:
        if ch == '(': depth += 1
        if ch == ')': depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip(): parts.append(cur.strip())
    return parts

def proto(n):
    m = re.search(r'extern\s+CL_API_ENTRY\s+([\w\s\*]+?)\s+CL_API_CALL\s+' + n +
                  r'\s*\((.*?)\)\s*CL_API_SUFFIX\w*\s*;', src, re.S)
    if not m: raise SystemExit('no prototype for ' + n)
    ret = ' '.join(m.group(1).split()); params = ' '.join(m.group(2).split())
    ps = split_params(params)
    if ps == ['void']: ps = []
    names = []
    for p in ps:
        fm = re.search(r'\(\s*CL_CALLBACK\s*\*\s*(\w+)\s*\)', p)
        names.append(fm.group(1) if fm else re.findall(r'(\w+)\s*(?:\[\s*\w*\s*\])?$', p)[0])
    return ret, params, names

out = []
# Typedefs + resolvers for the real driver functions the hand-written code needs
for n in sorted(set(EXTRA_DRIVER) | HAND_WRITTEN):
    ret, params, _ = proto(n)
    out.append('typedef %s (CL_API_CALL *real_%s_t)(%s);' % (ret, n, params))
    out.append('static real_%s_t real_%s(void) { static real_%s_t f = NULL; static int t = 0; '
               'if (!t) { f = (real_%s_t)shim_sym("%s"); t = 1; } return f; }' % (n, n, n, n, n))
out.append('')

for n in used:
    if n in HAND_WRITTEN: continue
    ret, params, names = proto(n)
    call = ', '.join(names)
    if ret == 'cl_int':
        fail = 'CL_PLATFORM_NOT_FOUND_KHR' if n == 'clGetPlatformIDs' else 'CL_INVALID_PLATFORM'
        pre = 'if (num_platforms) *num_platforms = 0; ' if n == 'clGetPlatformIDs' else ''
        body = '    if (!fn) { %sreturn %s; }\n    return fn(%s);' % (pre, fail, call)
    else:
        pre = 'if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM; ' if 'errcode_ret' in names else ''
        body = '    if (!fn) { %sreturn NULL; }\n    return fn(%s);' % (pre, call)
    out.append('CL_API_ENTRY %s CL_API_CALL %s(%s) {\n    typedef %s (CL_API_CALL *fn_t)(%s);\n'
               '    static fn_t fn = NULL; static int tried = 0;\n'
               '    if (!tried) { fn = (fn_t)shim_sym("%s"); tried = 1; }\n%s\n}\n'
               % (ret, n, params, ret, params, n, body))

template = open(__file__.replace('gen_opencl_shim.py', 'opencl_shim_template.c')).read()
open(out_path, 'w').write(template.replace('/*@GENERATED@*/', '\n'.join(out)))
print('generated %d forwarded functions (+%d hand-written)' % (len([n for n in used if n not in HAND_WRITTEN]), len(HAND_WRITTEN & set(used))))
