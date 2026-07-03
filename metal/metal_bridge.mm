#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "metal_types.h"

/* -----------------------------------------------------------------------
 * File-scope Metal objects retained for the program lifetime.
 * Mirrors the module-level device arrays in gpu_manager.cuf.
 * ----------------------------------------------------------------------- */
static id<MTLDevice>               s_device       = nil;
static id<MTLCommandQueue>         s_queue        = nil;
static id<MTLLibrary>              s_library      = nil;
static id<MTLBuffer>               s_uold         = nil;
static id<MTLBuffer>               s_unew         = nil;
static id<MTLBuffer>               s_grid         = nil;
static id<MTLBuffer>               s_nbor         = nil;
static id<MTLBuffer>               s_hash_key     = nil;
static id<MTLBuffer>               s_hash_val     = nil;
static id<MTLComputePipelineState> s_pso_set_unew = nil;
static id<MTLComputePipelineState> s_pso_set_uold = nil;
static id<MTLComputePipelineState> s_pso_cmpdt    = nil;
static id<MTLComputePipelineState> s_pso_godunov      = nil;
static id<MTLComputePipelineState> s_pso_insert_hash  = nil;
static id<MTLComputePipelineState> s_pso_build_nbor   = nil;

/* ----------------------------------------------------------------------- */
static id<MTLComputePipelineState> make_pso(NSString *name)
{
    NSError *error = nil;
    id<MTLFunction> fn = [s_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "[metal] kernel not found: %s\n", [name UTF8String]);
        exit(1);
    }
    id<MTLComputePipelineState> pso =
        [s_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
        fprintf(stderr, "[metal] pipeline error for %s: %s\n",
                [name UTF8String], [[error localizedDescription] UTF8String]);
        exit(1);
    }
    return pso;
}

/* -----------------------------------------------------------------------
 * mtl_init — create device, queue, load metallib, build pipeline states.
 * Mirrors mdl_initialize / cudaSetDevice in the CUDA path.
 * The metallib is expected next to the running executable (bin/).
 * ----------------------------------------------------------------------- */
extern "C" void mtl_init(void)
{
    s_device = MTLCreateSystemDefaultDevice();
    if (!s_device) {
        fprintf(stderr, "[metal] no Metal device found\n");
        exit(1);
    }
    s_queue = [s_device newCommandQueue];

    NSString *exec_path = [[[NSProcessInfo processInfo] arguments] firstObject];
    NSString *exec_dir  = [exec_path stringByDeletingLastPathComponent];
    NSString *lib_path  = [exec_dir stringByAppendingPathComponent:
                           @"ramses_kernels.metallib"];
    NSURL    *lib_url   = [NSURL fileURLWithPath:lib_path];

    NSError *error = nil;
    s_library = [s_device newLibraryWithURL:lib_url error:&error];
    if (!s_library) {
        fprintf(stderr, "[metal] cannot load %s: %s\n",
                [lib_path UTF8String],
                [[error localizedDescription] UTF8String]);
        exit(1);
    }

    s_pso_set_unew     = make_pso(@"set_unew_kernel");
    s_pso_set_uold     = make_pso(@"set_uold_kernel");
    s_pso_cmpdt        = make_pso(@"cmpdt_kernel");
    s_pso_godunov      = make_pso(@"hydro_integrator_kernel");
    s_pso_insert_hash  = make_pso(@"insert_hash_kernel");
    s_pso_build_nbor   = make_pso(@"build_nbor_kernel");

    fprintf(stdout, " Launching METAL.\n");
    fprintf(stdout, " Device name:       %s\n", [[s_device name] UTF8String]);
    fprintf(stdout, " Unified memory:    %s\n",
            s_device.hasUnifiedMemory ? "yes" : "no");
    fprintf(stdout, " Max working set:   %llu MB\n",
            (unsigned long long)s_device.recommendedMaxWorkingSetSize / (1024*1024));
    fprintf(stdout, " Max buffer length: %llu MB\n",
            (unsigned long long)s_device.maxBufferLength / (1024*1024));
    fprintf(stdout, " Max threads/tg:    %lu\n",
            (unsigned long)s_device.maxThreadsPerThreadgroup.width);
}

/* -----------------------------------------------------------------------
 * mtl_alloc_amr — allocate Metal-owned buffers for uold, unew, grid.
 * Mirrors gpu_allocate_amr in gpu_manager.cuf.
 * MTLResourceStorageModeShared: buffer lives in CPU/GPU shared DRAM.
 * Data is copied from the Fortran arrays in mtl_set_grid_device.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_alloc_amr(int ngridmax, int nvar, int twotondim, int hash_size)
{
    NSUInteger u_bytes        = (NSUInteger)ngridmax  * nvar * twotondim * sizeof(float);
    NSUInteger grid_bytes     = (NSUInteger)ngridmax  * sizeof(oct_t);
    NSUInteger nbor_bytes     = (NSUInteger)ngridmax  * 27   * sizeof(int);
    NSUInteger hash_key_bytes = (NSUInteger)hash_size * sizeof(long);
    NSUInteger hash_val_bytes = (NSUInteger)hash_size * sizeof(int);

    s_uold     = [s_device newBufferWithLength:u_bytes
                                       options:MTLResourceStorageModeShared];
    s_unew     = [s_device newBufferWithLength:u_bytes
                                       options:MTLResourceStorageModeShared];
    s_grid     = [s_device newBufferWithLength:grid_bytes
                                       options:MTLResourceStorageModeShared];
    s_nbor     = [s_device newBufferWithLength:nbor_bytes
                                       options:MTLResourceStorageModeShared];
    s_hash_key = [s_device newBufferWithLength:hash_key_bytes
                                       options:MTLResourceStorageModeShared];
    s_hash_val = [s_device newBufferWithLength:hash_val_bytes
                                       options:MTLResourceStorageModeShared];
}

/* -----------------------------------------------------------------------
 * mtl_set_grid_device — copy host arrays into Metal buffers (H->D).
 * Mirrors the cudaMemcpy calls in r_set_grid_device (gpu_manager.cuf).
 * On Apple Silicon the memcpy stays within DRAM (no PCIe), but is still
 * needed because the Metal buffer and the Fortran array are distinct
 * allocations at different addresses.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_set_grid_device(void *uold_ptr, void *unew_ptr,
                                    void *grid_ptr,
                                    int ngridmax, int nvar, int twotondim)
{
    size_t u_bytes    = (size_t)ngridmax * nvar * twotondim * sizeof(float);
    size_t grid_bytes = (size_t)ngridmax * sizeof(oct_t);
    memcpy(s_uold.contents, uold_ptr, u_bytes);
    memcpy(s_unew.contents, unew_ptr, u_bytes);
    memcpy(s_grid.contents, grid_ptr, grid_bytes);
}

/* -----------------------------------------------------------------------
 * mtl_transfer_grid_host — copy Metal uold buffer back to host (D->H).
 * Mirrors the cudaMemcpy calls in r_transfer_grid_host (gpu_manager.cuf).
 * Called before each output dump so m%uold reflects the GPU result.
 * Only uold is needed for I/O; grid is static for levelmin==levelmax.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_transfer_grid_host(void *uold_ptr,
                                       int ngridmax, int nvar, int twotondim)
{
    size_t u_bytes = (size_t)ngridmax * nvar * twotondim * sizeof(float);
    memcpy(uold_ptr, s_uold.contents, u_bytes);
}

/* -----------------------------------------------------------------------
 * mtl_device_sync — block until all previously submitted Metal work completes.
 * Mirrors cudaDeviceSynchronize() in the CUDA path.
 * An empty command buffer committed to the queue is sufficient: Metal
 * serialises command buffers in submission order, so waiting on this empty
 * one guarantees all prior dispatches have finished.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_device_sync(void)
{
    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_insert_hash — clear hash table and insert all oct Hilbert keys.
 * Mirrors the insert_hash_kernel<<<>>> call in r_set_grid_device
 * (gpu_manager.cuf).  The hash table is a reusable device structure;
 * keeping this separate from mtl_build_nbor allows it to be called
 * independently (e.g. after refinement) without rebuilding nbor.
 *
 * hash_key is zeroed first (empty-slot sentinel = 0); memset is safe
 * because MTLResourceStorageModeShared is CPU-accessible.
 * Thread layout: 128 threads/threadgroup — mirrors CUDA num_threads=128.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_insert_hash(int head_idx, int num_octs,
                                 int hash_size,
                                 int ckey_max_l, long key_off_l)
{
    memset(s_hash_key.contents, 0, s_hash_key.length);
    memset(s_hash_val.contents, 0, s_hash_val.length);

    NSUInteger tg128  = 128;
    MTLSize tg_size   = {tg128, 1, 1};
    MTLSize grid_size = {((NSUInteger)num_octs + tg128 - 1) / tg128, 1, 1};

    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_insert_hash];
    [enc setBuffer:s_grid     offset:0 atIndex:0];
    [enc setBuffer:s_hash_key offset:0 atIndex:1];
    [enc setBuffer:s_hash_val offset:0 atIndex:2];
    [enc setBytes:&hash_size  length:sizeof(int)  atIndex:3];
    [enc setBytes:&ckey_max_l length:sizeof(int)  atIndex:4];
    [enc setBytes:&key_off_l  length:sizeof(long) atIndex:5];
    [enc setBytes:&head_idx   length:sizeof(int)  atIndex:6];
    [enc setBytes:&num_octs   length:sizeof(int)  atIndex:7];
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_build_nbor — build the device nbor array from the already-populated
 * hash table by dispatching build_nbor_kernel.
 * Mirrors the 27-launch update_nbor_array loop in r_set_grid_device
 * (gpu_manager.cuf); a single dispatch replaces those 27 launches.
 * Thread layout: 128 threads/threadgroup.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_build_nbor(int head_idx, int num_subgrids,
                                int hash_size,
                                int ckey_max_l,    long key_off_l,
                                int *box_ckey_min, int *box_ckey_max,
                                int *periodic)
{
    NSUInteger tg128  = 128;
    MTLSize tg_size   = {tg128, 1, 1};
    MTLSize grid_size = {((NSUInteger)num_subgrids + tg128 - 1) / tg128, 1, 1};

    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_build_nbor];
    [enc setBuffer:s_grid     offset:0 atIndex:0];
    [enc setBuffer:s_nbor     offset:0 atIndex:1];
    [enc setBuffer:s_hash_key offset:0 atIndex:2];
    [enc setBuffer:s_hash_val offset:0 atIndex:3];
    [enc setBytes:&hash_size      length:sizeof(int)      atIndex:4];
    [enc setBytes:&ckey_max_l     length:sizeof(int)      atIndex:5];
    [enc setBytes:&key_off_l      length:sizeof(long)     atIndex:6];
    [enc setBytes:box_ckey_min    length:3 * sizeof(int)  atIndex:7];
    [enc setBytes:box_ckey_max    length:3 * sizeof(int)  atIndex:8];
    [enc setBytes:periodic        length:3 * sizeof(int)  atIndex:9];
    [enc setBytes:&head_idx       length:sizeof(int)      atIndex:10];
    [enc setBytes:&num_subgrids   length:sizeof(int)      atIndex:11];
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_set_unew — dispatch set_unew_kernel: unew = uold for octs at ilevel.
 * Thread layout mirrors CUDA: dim3(8, 16, 1) per threadgroup,
 * ceil(num_octs / 16) threadgroups.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_set_unew(int head_idx, int num_octs)
{
    MTLSize tg_size   = {8, 16, 1};
    MTLSize grid_size = {((NSUInteger)num_octs + 15) / 16, 1, 1};

    id<MTLCommandBuffer>        cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_set_unew];
    [enc setBuffer:s_uold   offset:0 atIndex:0];
    [enc setBuffer:s_unew   offset:0 atIndex:1];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_set_uold — dispatch set_uold_kernel: uold = unew for octs at ilevel.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_set_uold(int head_idx, int num_octs)
{
    MTLSize tg_size   = {8, 16, 1};
    MTLSize grid_size = {((NSUInteger)num_octs + 15) / 16, 1, 1};

    id<MTLCommandBuffer>        cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_set_uold];
    [enc setBuffer:s_uold   offset:0 atIndex:0];
    [enc setBuffer:s_unew   offset:0 atIndex:1];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_cmpdt — dispatch cmpdt_kernel and read back results.
 * Mirrors gpu_cmpdt in gpu_runner.cuf.
 *
 * data_buf layout: atomic_uint[5] reinterpreted as float[5] on readback.
 *   [0..3] fp32 accumulated via CAS atomic_add_float
 *   [4]    fp32 min via uint bit-cast atomic_min_float_bits
 *
 * 256 threads/threadgroup; SIMD reduction inside the kernel collapses to
 * one atomic write per threadgroup.  dispatchThreadgroups with
 * ceil(num_octs*8 / 256) threadgroups.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_cmpdt(int head_idx, int num_octs,
                           float dx, float gamma, float smallr, float smallc2,
                           float courant_factor, float *constant_gravity,
                           float *mass, float *ekin, float *eint, float *emag,
                           float *dt)
{
    float dt_init = courant_factor * dx / sqrtf(smallc2);
    /* data_buf layout: atomic_uint[5] = {mass, ekin, eint, emag, dt}
     * [0..3] initialised to 0 (float bit-pattern); [4] initialised to dt_init. */
    uint32_t h_data[5] = {0, 0, 0, 0, 0};
    memcpy(&h_data[4], &dt_init, sizeof(float));
    id<MTLBuffer> data_buf =
        [s_device newBufferWithBytes:h_data
                              length:5 * sizeof(uint32_t)
                             options:MTLResourceStorageModeShared];

    float cg[3] = {constant_gravity[0], constant_gravity[1], constant_gravity[2]};

    NSUInteger total_cells = (NSUInteger)num_octs * 8;   /* 8 = twotondim for NDIM=3 */
    MTLSize tg_size   = {1024, 1, 1};
    MTLSize grid_size = {(total_cells + 1023) / 1024, 1, 1};

    id<MTLCommandBuffer>        cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_cmpdt];
    [enc setBuffer:s_grid    offset:0 atIndex:0];
    [enc setBuffer:s_uold    offset:0 atIndex:1];
    [enc setBuffer:data_buf  offset:0 atIndex:2];
    [enc setBytes:&head_idx       length:sizeof(int)       atIndex:3];
    [enc setBytes:&num_octs       length:sizeof(int)       atIndex:4];
    [enc setBytes:&dx             length:sizeof(float)     atIndex:5];
    [enc setBytes:&gamma          length:sizeof(float)     atIndex:6];
    [enc setBytes:&smallr         length:sizeof(float)     atIndex:7];
    [enc setBytes:&smallc2        length:sizeof(float)     atIndex:8];
    [enc setBytes:&courant_factor length:sizeof(float)     atIndex:9];
    [enc setBytes:cg              length:3 * sizeof(float) atIndex:10];

    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    float *result = (float *)data_buf.contents;
    *mass = result[0];
    *ekin = result[1];
    *eint = result[2];
    *emag = result[3];
    *dt   = result[4];
}

/* -----------------------------------------------------------------------
 * mtl_godunov — dispatch hydro_integrator_kernel (MUSCL-Hancock).
 * Mirrors gpu_godunov in gpu_runner.cuf.
 * Thread layout mirrors CUDA nsubgrid=1: 64 threads/threadgroup,
 * 1 threadgroup per oct (subgrid).
 * ----------------------------------------------------------------------- */
extern "C" void mtl_godunov(int head_idx, int num_subgrids, int ngridmax,
                             int ilevel, int levelmin, int levelmax,
                             float gamma, float smallr, float smallc2,
                             float dt, float dx, int slope, int riemann,
                             float *constant_gravity)
{
    float cg[3] = {constant_gravity[0], constant_gravity[1], constant_gravity[2]};

    /* 64 threads/threadgroup matches CUDA nsubgrid=1 branch */
    MTLSize tg_size   = {64, 1, 1};
    MTLSize grid_size = {(NSUInteger)num_subgrids, 1, 1};

    id<MTLCommandBuffer>        cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_godunov];
    [enc setBuffer:s_grid  offset:0 atIndex:0];
    [enc setBuffer:s_uold  offset:0 atIndex:1];
    [enc setBuffer:s_unew  offset:0 atIndex:2];
    [enc setBuffer:s_nbor  offset:0 atIndex:3];
    [enc setBytes:&head_idx     length:sizeof(int)       atIndex:4];
    [enc setBytes:&num_subgrids length:sizeof(int)       atIndex:5];
    [enc setBytes:&ngridmax     length:sizeof(int)       atIndex:6];
    [enc setBytes:&ilevel       length:sizeof(int)       atIndex:7];
    [enc setBytes:&levelmin     length:sizeof(int)       atIndex:8];
    [enc setBytes:&levelmax     length:sizeof(int)       atIndex:9];
    [enc setBytes:&gamma        length:sizeof(float)     atIndex:10];
    [enc setBytes:&smallr       length:sizeof(float)     atIndex:11];
    [enc setBytes:&smallc2      length:sizeof(float)     atIndex:12];
    [enc setBytes:&dt           length:sizeof(float)     atIndex:13];
    [enc setBytes:&dx           length:sizeof(float)     atIndex:14];
    [enc setBytes:&slope        length:sizeof(int)       atIndex:15];
    [enc setBytes:&riemann      length:sizeof(int)       atIndex:16];
    [enc setBytes:cg            length:3 * sizeof(float) atIndex:17];
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}
