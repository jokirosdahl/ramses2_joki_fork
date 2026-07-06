#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <mach/mach.h>

/* Returns resident set size in pages (4096 bytes each), matching Linux /proc/self/stat field 24. */
extern "C" long getmem_mac(void)
{
    struct task_basic_info info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0L;
    return (long)(info.resident_size / 4096);
}

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
/* AMR refinement buffers (allocated by mtl_alloc_refine) */
static id<MTLBuffer>               s_flag1            = nil;
static id<MTLBuffer>               s_flag2            = nil;
static id<MTLBuffer>               s_father           = nil;
static id<MTLBuffer>               s_swap_local       = nil;
static id<MTLBuffer>               s_swap_global      = nil;
static id<MTLBuffer>               s_prefix_sum       = nil;
static id<MTLBuffer>               s_partial_sums     = nil;  /* level-1 scratch (ps0): ceil(n/256)      */
static id<MTLBuffer>               s_partial_sums_2   = nil;  /* level-2 scratch (ps1): ceil(n/256^2)    */
static id<MTLBuffer>               s_partial_sums_3   = nil;  /* level-3 scratch (ps2): ceil(n/256^3)    */
static id<MTLBuffer>               s_partial_sums_4   = nil;  /* dummy sink for deepest single-block pass */
static id<MTLBuffer>               s_ifree_dev        = nil;
static id<MTLBuffer>               s_ifree_cache_dev  = nil;
static id<MTLBuffer>               s_ckey_max_dev     = nil;
static id<MTLBuffer>               s_key_off_dev      = nil;
static id<MTLBuffer>               s_box_ckey_min_dev = nil;
static id<MTLBuffer>               s_box_ckey_max_dev = nil;
static id<MTLBuffer>               s_periodic_dev     = nil;

/* Fence for inter-encoder ordering within a single command buffer.
 * Used by mtl_hilbert_sort_level to synchronize between 6 encoders per pass
 * without needing 6 separate commit/waitUntilCompleted cycles.            */
static id<MTLFence>                s_sort_fence       = nil;
static id<MTLBuffer>               s_count_buf        = nil;  /* 1×int, persistent for batched flag reductions */

static id<MTLComputePipelineState> s_pso_set_unew = nil;
static id<MTLComputePipelineState> s_pso_set_uold = nil;
static id<MTLComputePipelineState> s_pso_cmpdt    = nil;
static id<MTLComputePipelineState> s_pso_godunov      = nil;
static id<MTLComputePipelineState> s_pso_insert_hash  = nil;
static id<MTLComputePipelineState> s_pso_build_nbor   = nil;
static id<MTLComputePipelineState> s_pso_scan_block        = nil;
static id<MTLComputePipelineState> s_pso_scan_fixup        = nil;
static id<MTLComputePipelineState> s_pso_reset_flag1       = nil;
static id<MTLComputePipelineState> s_pso_reset_flag2       = nil;
static id<MTLComputePipelineState> s_pso_init_flag         = nil;
static id<MTLComputePipelineState> s_pso_count_flag1       = nil;
static id<MTLComputePipelineState> s_pso_hydro_flag        = nil;
static id<MTLComputePipelineState> s_pso_count_neighbors   = nil;
static id<MTLComputePipelineState> s_pso_flag_count        = nil;
static id<MTLComputePipelineState> s_pso_enforce_rules     = nil;
static id<MTLComputePipelineState> s_pso_update_father     = nil;

/* PSOs for AMR refine/sort/cache kernels (refine.metal) */
static id<MTLComputePipelineState> s_pso_refine            = nil;
static id<MTLComputePipelineState> s_pso_derefine          = nil;
static id<MTLComputePipelineState> s_pso_free_hash         = nil;
static id<MTLComputePipelineState> s_pso_update_hash       = nil;
static id<MTLComputePipelineState> s_pso_insert_hash_all   = nil;
static id<MTLComputePipelineState> s_pso_init_swap_table   = nil;
static id<MTLComputePipelineState> s_pso_prefix_level      = nil;
static id<MTLComputePipelineState> s_pso_prefix_bit        = nil;
static id<MTLComputePipelineState> s_pso_local_swap        = nil;
static id<MTLComputePipelineState> s_pso_global_swap       = nil;
static id<MTLComputePipelineState> s_pso_gather_grid       = nil;
static id<MTLComputePipelineState> s_pso_scatter_grid      = nil;
static id<MTLComputePipelineState> s_pso_gather_flag       = nil;
static id<MTLComputePipelineState> s_pso_scatter_flag      = nil;
static id<MTLComputePipelineState> s_pso_gather_hydro      = nil;
static id<MTLComputePipelineState> s_pso_nbor_prefix       = nil;
static id<MTLComputePipelineState> s_pso_cache_swap        = nil;
static id<MTLComputePipelineState> s_pso_make_cache        = nil;
static id<MTLComputePipelineState> s_pso_insert_hash_cache = nil;
static id<MTLComputePipelineState> s_pso_upload            = nil;

static int s_nvar = 5;   /* set in mtl_alloc_amr; used in mtl_blit_unew_to_uold */

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
    s_pso_scan_block        = make_pso(@"scan_block_kernel");
    s_pso_scan_fixup        = make_pso(@"scan_fixup_kernel");
    s_pso_reset_flag1       = make_pso(@"reset_flag1_kernel");
    s_pso_reset_flag2       = make_pso(@"reset_flag2_kernel");
    s_pso_init_flag         = make_pso(@"init_flag_kernel");
    s_pso_count_flag1       = make_pso(@"count_flag1_kernel");
    s_pso_hydro_flag        = make_pso(@"hydro_flag_kernel");
    s_pso_count_neighbors   = make_pso(@"count_neighbors_kernel");
    s_pso_flag_count        = make_pso(@"flag_count_kernel");
    s_pso_enforce_rules     = make_pso(@"enforce_rules_kernel");
    s_pso_update_father     = make_pso(@"update_father_kernel");

    s_pso_refine            = make_pso(@"refine_kernel");
    s_pso_derefine          = make_pso(@"derefine_kernel");
    s_pso_free_hash         = make_pso(@"free_hash_kernel");
    s_pso_update_hash       = make_pso(@"update_hash_kernel");
    s_pso_insert_hash_all   = make_pso(@"insert_hash_all_kernel");
    s_pso_init_swap_table   = make_pso(@"init_global_swap_table_kernel");
    s_pso_prefix_level      = make_pso(@"init_prefix_sum_level_kernel");
    s_pso_prefix_bit        = make_pso(@"init_prefix_sum_bit_kernel");
    s_pso_local_swap        = make_pso(@"compute_local_swap_table_kernel");
    s_pso_global_swap       = make_pso(@"update_global_swap_table_kernel");
    s_pso_gather_grid       = make_pso(@"sort_gather_grid_kernel");
    s_pso_scatter_grid      = make_pso(@"sort_scatter_grid_kernel");
    s_pso_gather_flag       = make_pso(@"sort_gather_flag_kernel");
    s_pso_scatter_flag      = make_pso(@"sort_scatter_flag_kernel");
    s_pso_gather_hydro      = make_pso(@"sort_gather_hydro_kernel");
    s_pso_nbor_prefix       = make_pso(@"update_nbor_prefix_kernel");
    s_pso_cache_swap        = make_pso(@"compute_cache_swap_table_kernel");
    s_pso_make_cache        = make_pso(@"make_cache_octs_kernel");
    s_pso_insert_hash_cache = make_pso(@"insert_hash_cache_kernel");
    s_pso_upload            = make_pso(@"upload_kernel");

    fprintf(stdout, " Launching METAL.\n");
    s_sort_fence = [s_device newFence];
    s_count_buf  = [s_device newBufferWithLength:sizeof(int)
                                         options:MTLResourceStorageModeShared];
    *(int *)s_count_buf.contents = 0;

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
/* ncachemax is included so grid, nbor, and the hydro arrays cover the full
 * [1..ngridmax+ncachemax] range required by AMR ghost-zone caching.
 * For the unigrid PoC (ncachemax>0 but cache never populated) the extra
 * allocation is harmless — behaviour identical to the old 4-argument form. */
extern "C" void mtl_alloc_amr(int ngridmax, int ncachemax,
                               int nvar, int twotondim, int hash_size)
{
    s_nvar = nvar;
    int ntotal = ngridmax + ncachemax;
    NSUInteger u_bytes        = (NSUInteger)ntotal    * nvar * twotondim * sizeof(float);
    NSUInteger grid_bytes     = (NSUInteger)ntotal    * sizeof(oct_t);
    NSUInteger nbor_bytes     = (NSUInteger)ntotal    * 27   * sizeof(int);
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
 * mtl_upload_flag1 — copy host flag1(8,ngridmax) to device s_flag1.
 * Mirrors the CUDA path: `flag1 = pst%s%m%flag1` in gpu_manager.cuf.
 * Called from r_set_grid_device when nlevelmax > levelmin so that
 * derefine_kernel reads the correct refinement flags, not stale zeros.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_upload_flag1(void *flag1_host, int ngridmax)
{
    size_t nbytes = (size_t)8 * ngridmax * sizeof(int);
    memcpy(s_flag1.contents, flag1_host, nbytes);
}

/* -----------------------------------------------------------------------
 * mtl_transfer_grid_host — copy Metal uold buffer back to host (D->H).
 * Mirrors the cudaMemcpy calls in r_transfer_grid_host (gpu_manager.cuf).
 * Called before each output dump so m%uold reflects the GPU result.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_transfer_grid_host(void *uold_ptr,
                                       int ngridmax, int nvar, int twotondim)
{
    size_t u_bytes = (size_t)ngridmax * nvar * twotondim * sizeof(float);
    memcpy(uold_ptr, s_uold.contents, u_bytes);
}

/* -----------------------------------------------------------------------
 * mtl_transfer_grid_struct_host — copy Metal s_grid buffer back to host.
 * Required for AMR runs (levelmin < levelmax): metal_refine reorders octs
 * via Hilbert sort + scatter, updating s_grid on the device.  Without this
 * readback, output_amr reads stale host ckey/refined values and amr2map
 * produces a garbled level/density map.
 * Only the first ngridmax slots are written (cache octs start at ngridmax+1
 * and are never referenced by the output routines).
 * ----------------------------------------------------------------------- */
extern "C" void mtl_transfer_grid_struct_host(void *grid_ptr, int ngridmax)
{
    size_t grid_bytes = (size_t)ngridmax * sizeof(oct_t);
    memcpy(grid_ptr, s_grid.contents, grid_bytes);
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
    [enc setBuffer:s_father     offset:0               atIndex:18];
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_upload — dispatch upload_kernel (restriction: fine → coarse level).
 * Mirrors gpu_upload in gpu_runner.cuf.
 * One thread per fine oct (128 threads/threadgroup).
 * ----------------------------------------------------------------------- */
extern "C" void mtl_upload(int head_idx, int num_octs,
                            int internal_energy,
                            float gamma, float smallr, float smallc2)
{
    if (num_octs <= 0) return;
    NSUInteger tg_size = 128;
    NSUInteger num_tg  = ((NSUInteger)num_octs + tg_size - 1) / tg_size;

    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_upload];
    [enc setBuffer:s_grid          offset:0 atIndex:0];
    [enc setBuffer:s_father        offset:0 atIndex:1];
    [enc setBuffer:s_uold          offset:0 atIndex:2];
    [enc setBytes:&head_idx        length:sizeof(int)   atIndex:3];
    [enc setBytes:&num_octs        length:sizeof(int)   atIndex:4];
    [enc setBytes:&internal_energy length:sizeof(int)   atIndex:5];
    [enc setBytes:&gamma           length:sizeof(float) atIndex:6];
    [enc setBytes:&smallr          length:sizeof(float) atIndex:7];
    [enc setBytes:&smallc2         length:sizeof(float) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_alloc_refine — allocate device buffers for AMR refinement, sorting,
 * ghost-zone cache, and per-level Hilbert parameters.
 * Mirrors gpu_allocate_amr flag/sort arrays in gpu_manager.cuf.
 * All buffers are MTLResourceStorageModeShared (unified memory) and zeroed.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_alloc_refine(int ngridmax, int ncachemax, int nlevelmax)
{
    int ntotal    = ngridmax + ncachemax;
    int npartial  = (ntotal    + 255) / 256;  /* ps0: ceil(n/256)   */
    int npartial2 = (npartial  + 255) / 256;  /* ps1: ceil(n/256^2) */
    int npartial3 = (npartial2 + 255) / 256;  /* ps2: ceil(n/256^3) */
    int nlevels   = nlevelmax + 1;

    NSUInteger flag_bytes     = (NSUInteger)8 * ntotal   * sizeof(int);  /* 8 = twotondim NDIM=3 */
    NSUInteger oct_bytes      = (NSUInteger)ntotal        * sizeof(int);  /* father/swap/prefix */
    NSUInteger partial_bytes  = (NSUInteger)npartial      * sizeof(int);
    NSUInteger partial2_bytes = (NSUInteger)npartial2     * sizeof(int);
    NSUInteger partial3_bytes = (NSUInteger)npartial3     * sizeof(int);
    NSUInteger partial4_bytes = sizeof(int);                              /* 1-element dummy sink */
    NSUInteger scalar_bytes  = sizeof(int);
    NSUInteger ckey_bytes    = (NSUInteger)nlevels      * sizeof(int);
    NSUInteger koff_bytes    = (NSUInteger)nlevels      * sizeof(long);
    NSUInteger box_bytes     = (NSUInteger)3 * nlevels  * sizeof(int);
    NSUInteger per_bytes     = 3 * sizeof(int);

    s_flag1        = [s_device newBufferWithLength:flag_bytes    options:MTLResourceStorageModeShared];
    s_flag2        = [s_device newBufferWithLength:flag_bytes    options:MTLResourceStorageModeShared];
    s_father       = [s_device newBufferWithLength:oct_bytes     options:MTLResourceStorageModeShared];
    s_swap_local   = [s_device newBufferWithLength:oct_bytes     options:MTLResourceStorageModeShared];
    s_swap_global  = [s_device newBufferWithLength:oct_bytes     options:MTLResourceStorageModeShared];
    s_prefix_sum   = [s_device newBufferWithLength:oct_bytes     options:MTLResourceStorageModeShared];
    s_partial_sums   = [s_device newBufferWithLength:partial_bytes  options:MTLResourceStorageModeShared];
    s_partial_sums_2 = [s_device newBufferWithLength:partial2_bytes options:MTLResourceStorageModeShared];
    s_partial_sums_3 = [s_device newBufferWithLength:partial3_bytes options:MTLResourceStorageModeShared];
    s_partial_sums_4 = [s_device newBufferWithLength:partial4_bytes options:MTLResourceStorageModeShared];
    s_ifree_dev        = [s_device newBufferWithLength:scalar_bytes options:MTLResourceStorageModeShared];
    s_ifree_cache_dev  = [s_device newBufferWithLength:scalar_bytes options:MTLResourceStorageModeShared];
    s_ckey_max_dev     = [s_device newBufferWithLength:ckey_bytes   options:MTLResourceStorageModeShared];
    s_key_off_dev      = [s_device newBufferWithLength:koff_bytes   options:MTLResourceStorageModeShared];
    s_box_ckey_min_dev = [s_device newBufferWithLength:box_bytes    options:MTLResourceStorageModeShared];
    s_box_ckey_max_dev = [s_device newBufferWithLength:box_bytes    options:MTLResourceStorageModeShared];
    s_periodic_dev     = [s_device newBufferWithLength:per_bytes    options:MTLResourceStorageModeShared];

    memset(s_flag1.contents,           0, flag_bytes);
    memset(s_flag2.contents,           0, flag_bytes);
    memset(s_father.contents,          0, oct_bytes);
    memset(s_swap_local.contents,      0, oct_bytes);
    memset(s_swap_global.contents,     0, oct_bytes);
    memset(s_prefix_sum.contents,      0, oct_bytes);
    memset(s_partial_sums.contents,    0, partial_bytes);
    memset(s_partial_sums_2.contents,  0, partial2_bytes);
    memset(s_partial_sums_3.contents,  0, partial3_bytes);
    memset(s_partial_sums_4.contents,  0, partial4_bytes);
    memset(s_ifree_dev.contents,       0, scalar_bytes);
    memset(s_ifree_cache_dev.contents, 0, scalar_bytes);
}

/* -----------------------------------------------------------------------
 * mtl_upload_level_params — copy per-level Hilbert parameters from host to
 * device.  Called once from metal_allocate_amr after init_amr populates them.
 *
 * box_ckey_min/max are Fortran (ndim, nlevelmax+1) column-major arrays;
 * the C pointer receives them in the same byte order, so kernels index as
 * box_ckey_min_dev[3*(lev-1) + d] (0-based) to get box_ckey_min(d+1, lev).
 * ----------------------------------------------------------------------- */
extern "C" void mtl_upload_level_params(void *ckey_max, void *key_off,
                                         void *box_ckey_min, void *box_ckey_max,
                                         int  *periodic,    int nlevelmax)
{
    int nlevels = nlevelmax + 1;
    memcpy(s_ckey_max_dev.contents,     ckey_max,     nlevels * sizeof(int));
    memcpy(s_key_off_dev.contents,      key_off,      nlevels * sizeof(long));
    memcpy(s_box_ckey_min_dev.contents, box_ckey_min, 3 * nlevels * sizeof(int));
    memcpy(s_box_ckey_max_dev.contents, box_ckey_max, 3 * nlevels * sizeof(int));
    memcpy(s_periodic_dev.contents,     periodic,     3 * sizeof(int));
}

/* -----------------------------------------------------------------------
 * scan_phase — helper that dispatches one pass of scan_block_kernel or
 * scan_fixup_kernel over the given buffer range.
 * offset and n are 0-based (converted from Fortran 1-based by the caller).
 * ----------------------------------------------------------------------- */
static void scan_phase(id<MTLComputePipelineState> pso,
                       id<MTLBuffer> data, id<MTLBuffer> psums,
                       int offset, int n,
                       id<MTLCommandBuffer> cmd)
{
    NSUInteger tg   = 256;
    NSUInteger nblk = ((NSUInteger)n + tg - 1) / tg;
    MTLSize tg_size   = {tg, 1, 1};
    MTLSize grid_size = {nblk, 1, 1};

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:data  offset:0 atIndex:0];
    [enc setBuffer:psums offset:0 atIndex:1];
    [enc setBytes:&offset length:sizeof(int) atIndex:2];
    [enc setBytes:&n      length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [enc endEncoding];
}

/* -----------------------------------------------------------------------
 * mtl_prefix_scan — inclusive prefix scan of s_prefix_sum[offset..offset+n-1].
 * offset is 0-based (Fortran head_idx - 1).
 *
 * Three-phase approach (mirrors block_scan + scan(partial_sums) + uniform_add):
 *   Phase 1: scan_block_kernel  — within-block scans, block totals → s_partial_sums
 *   Phase 2: scan_block_kernel  — scan the partial_sums array (if n > 256)
 *   Phase 3: scan_fixup_kernel  — add partial_sums[bid-1] to block bid (if n > 256)
 *
 * All phases are submitted as separate command buffers and serialised via
 * waitUntilCompleted between phases (phase 2 needs phase 1 done; phase 3 needs
 * phase 2 done).
 * ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
 * mtl_prefix_scan — inclusive prefix scan of s_prefix_sum[offset..offset+n-1].
 *
 * Mirrors gpu_scan in gpu_runner.cuf exactly: three separate scratch buffers
 * (s_partial_sums / _2 / _3) hold block totals at successive levels so that
 * no scan pass ever aliases its data buffer with its partial-sums output.
 * s_partial_sums_4 is a 1-element dummy sink for the deepest single-block
 * pass (which must write its block total somewhere but the value is unused).
 *
 * Three cases, identical to the CUDA port:
 *   n ≤ 256^2 = 65,536          : 2-level (ps0 only)
 *   n ≤ 256^3 = 16,777,216      : 3-level (ps0, ps1)
 *   n ≤ INT_MAX                  : 4-level (ps0, ps1, ps2)
 * ----------------------------------------------------------------------- */
extern "C" void mtl_prefix_scan(int offset, int n)
{
    if (n <= 0) return;

    const int BS  = 256;
    int nb0 = (n   + BS - 1) / BS;
    int nb1 = (nb0 + BS - 1) / BS;
    int nb2 = (nb1 + BS - 1) / BS;

#define CMD_WAIT(body) do { \
    id<MTLCommandBuffer> _c = [s_queue commandBuffer]; \
    body; \
    [_c commit]; [_c waitUntilCompleted]; \
} while(0)

    if (n <= BS * BS) {
        /* ---- 2-level: n ≤ 65,536 ---------------------------------------- */
        CMD_WAIT(scan_phase(s_pso_scan_block, s_prefix_sum,   s_partial_sums,   offset, n,   _c));
        if (nb0 == 1) goto done;
        /* single-block scan of ps0; total → ps1[0] (unused dummy) */
        CMD_WAIT(scan_phase(s_pso_scan_block, s_partial_sums,   s_partial_sums_2, 0,      nb0, _c));
        CMD_WAIT(scan_phase(s_pso_scan_fixup, s_prefix_sum,   s_partial_sums,   offset, n,   _c));

    } else if (n <= BS * BS * BS) {
        /* ---- 3-level: n ≤ 16,777,216 ------------------------------------- */
        CMD_WAIT(scan_phase(s_pso_scan_block, s_prefix_sum,   s_partial_sums,   offset, n,   _c));
        /* scan ps0 with nb1 blocks; block totals → ps1 */
        CMD_WAIT(scan_phase(s_pso_scan_block, s_partial_sums,   s_partial_sums_2, 0,      nb0, _c));
        if (nb1 > 1) {
            /* single-block scan of ps1; total → ps2[0] (unused dummy) */
            CMD_WAIT(scan_phase(s_pso_scan_block, s_partial_sums_2, s_partial_sums_3, 0,      nb1, _c));
            /* fixup ps0 using ps1 */
            CMD_WAIT(scan_phase(s_pso_scan_fixup, s_partial_sums,   s_partial_sums_2, 0,      nb0, _c));
        }
        CMD_WAIT(scan_phase(s_pso_scan_fixup, s_prefix_sum,   s_partial_sums,   offset, n,   _c));

    } else {
        /* ---- 4-level: n ≤ INT_MAX ---------------------------------------- */
        CMD_WAIT(scan_phase(s_pso_scan_block, s_prefix_sum,     s_partial_sums,   offset, n,   _c));
        /* scan ps0 with nb1 blocks; block totals → ps1 */
        CMD_WAIT(scan_phase(s_pso_scan_block, s_partial_sums,   s_partial_sums_2, 0,      nb0, _c));
        /* scan ps1 with nb2 blocks; block totals → ps2 */
        CMD_WAIT(scan_phase(s_pso_scan_block, s_partial_sums_2, s_partial_sums_3, 0,      nb1, _c));
        if (nb2 > 1) {
            /* single-block scan of ps2; total → ps3[0] (unused dummy) */
            CMD_WAIT(scan_phase(s_pso_scan_block, s_partial_sums_3, s_partial_sums_4, 0,  nb2, _c));
            /* fixup ps1 using ps2 */
            CMD_WAIT(scan_phase(s_pso_scan_fixup, s_partial_sums_2, s_partial_sums_3, 0,  nb1, _c));
        }
        /* fixup ps0 using ps1 */
        CMD_WAIT(scan_phase(s_pso_scan_fixup, s_partial_sums,   s_partial_sums_2, 0,      nb0, _c));
        CMD_WAIT(scan_phase(s_pso_scan_fixup, s_prefix_sum,     s_partial_sums,   offset, n,   _c));
    }

done:;
#undef CMD_WAIT
}

/* -----------------------------------------------------------------------
 * mtl_get_prefix_total — read the inclusive sum of prefix_sum[offset..offset+n-1]
 * after mtl_prefix_scan.  Equivalent to get_total_sum in gpu_scan.cuf.
 * Returns prefix_sum[offset + n - 1] directly from shared memory.
 * ----------------------------------------------------------------------- */
extern "C" int mtl_get_prefix_total(int offset, int n)
{
    if (n <= 0) return 0;
    return ((int *)s_prefix_sum.contents)[offset + n - 1];
}

/* -----------------------------------------------------------------------
 * dispatch_2d_flag — helper for the 2D flag kernels (8 cells × 16 octs).
 * Mirrors CUDA <<<dim3(N,1,1), dim3(8,16,1)>>>.
 * ----------------------------------------------------------------------- */
static void dispatch_2d_flag(id<MTLComputePipelineState> pso,
                              id<MTLComputeCommandEncoder> enc,
                              int num_octs)
{
    NSUInteger tg128  = 128;   /* 8 × 16 = 128 */
    NSUInteger nblk   = ((NSUInteger)num_octs + 15) / 16;
    MTLSize tg_size   = {tg128, 1, 1};
    MTLSize grid_size = {nblk,  1, 1};
    [enc setComputePipelineState:pso];
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

/* -----------------------------------------------------------------------
 * mtl_reset_flag1 — zero flag1 for [head_idx .. head_idx+num_octs-1].
 * ----------------------------------------------------------------------- */
extern "C" void mtl_reset_flag1(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_reset_flag1];
    [enc setBuffer:s_flag1    offset:0 atIndex:0];
    [enc setBytes:&head_idx   length:sizeof(int) atIndex:1];
    [enc setBytes:&num_octs   length:sizeof(int) atIndex:2];
    NSUInteger nblk = ((NSUInteger)num_octs + 15) / 16;
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_reset_flag2 — zero flag2 for [head_idx .. head_idx+num_octs-1].
 * ----------------------------------------------------------------------- */
extern "C" void mtl_reset_flag2(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_reset_flag2];
    [enc setBuffer:s_flag2    offset:0 atIndex:0];
    [enc setBytes:&head_idx   length:sizeof(int) atIndex:1];
    [enc setBytes:&num_octs   length:sizeof(int) atIndex:2];
    NSUInteger nblk = ((NSUInteger)num_octs + 15) / 16;
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_init_flag — flag parent cell for each fine oct at ilevel+1.
 * head_idx / num_octs refer to the FINE level (ilevel+1).
 * ----------------------------------------------------------------------- */
extern "C" void mtl_init_flag(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg128  = 128;
    NSUInteger nblk   = ((NSUInteger)num_octs + tg128 - 1) / tg128;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_init_flag];
    [enc setBuffer:s_flag1    offset:0 atIndex:0];
    [enc setBuffer:s_grid     offset:0 atIndex:1];
    [enc setBuffer:s_father   offset:0 atIndex:2];
    [enc setBytes:&head_idx   length:sizeof(int) atIndex:3];
    [enc setBytes:&num_octs   length:sizeof(int) atIndex:4];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg128,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_count_flag1 — reduce sum of flag1 cells; returns the count.
 * head_idx / num_octs refer to the coarse level (ilevel).
 * ----------------------------------------------------------------------- */
extern "C" int mtl_count_flag1(int head_idx, int num_octs)
{
    if (num_octs <= 0) return 0;
    /* Allocate a zeroed one-int buffer for the atomic result */
    int zero = 0;
    id<MTLBuffer> result_buf =
        [s_device newBufferWithBytes:&zero
                              length:sizeof(int)
                             options:MTLResourceStorageModeShared];

    NSUInteger tg1024 = 1024;
    NSUInteger total_cells = (NSUInteger)num_octs * 8;
    NSUInteger nblk  = (total_cells + tg1024 - 1) / tg1024;

    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_count_flag1];
    [enc setBuffer:s_flag1    offset:0 atIndex:0];
    [enc setBuffer:result_buf offset:0 atIndex:1];
    [enc setBytes:&head_idx   length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs   length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg1024,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    int count = *(int *)result_buf.contents;
    return count;
}

/* -----------------------------------------------------------------------
 * mtl_hydro_flag — gradient density/pressure refinement criterion.
 * head_idx / num_octs: octs at ilevel.  No MHD, no GRAV.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_hydro_flag(int head_idx, int num_octs,
                                float gamma, float smallr, float smallc2,
                                float err_grad_d, float err_grad_p,
                                float floor_d,   float floor_p)
{
    if (num_octs <= 0) return;
    NSUInteger nblk = ((NSUInteger)num_octs + 15) / 16;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_hydro_flag];
    [enc setBuffer:s_flag1      offset:0 atIndex:0];
    [enc setBuffer:s_nbor       offset:0 atIndex:1];
    [enc setBuffer:s_uold       offset:0 atIndex:2];
    [enc setBytes:&head_idx     length:sizeof(int)   atIndex:3];
    [enc setBytes:&num_octs     length:sizeof(int)   atIndex:4];
    [enc setBytes:&gamma        length:sizeof(float) atIndex:5];
    [enc setBytes:&smallr       length:sizeof(float) atIndex:6];
    [enc setBytes:&smallc2      length:sizeof(float) atIndex:7];
    [enc setBytes:&err_grad_d   length:sizeof(float) atIndex:8];
    [enc setBytes:&err_grad_p   length:sizeof(float) atIndex:9];
    [enc setBytes:&floor_d      length:sizeof(float) atIndex:10];
    [enc setBytes:&floor_p      length:sizeof(float) atIndex:11];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_count_neighbors — write flag2[cell,oct] = # flagged face neighbours.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_count_neighbors(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger nblk = ((NSUInteger)num_octs + 15) / 16;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_count_neighbors];
    [enc setBuffer:s_flag2    offset:0 atIndex:0];
    [enc setBuffer:s_flag1    offset:0 atIndex:1];
    [enc setBuffer:s_nbor     offset:0 atIndex:2];
    [enc setBytes:&head_idx   length:sizeof(int) atIndex:3];
    [enc setBytes:&num_octs   length:sizeof(int) atIndex:4];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_flag_count — promote flag1 if flag2 >= num_nbors; clear flag2 if
 * flag1 is already set.  num_nbors = n_nbor(idim) = {1,2,2} for idim=1..3.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_flag_count(int head_idx, int num_octs, int num_nbors)
{
    if (num_octs <= 0) return;
    NSUInteger nblk = ((NSUInteger)num_octs + 15) / 16;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_flag_count];
    [enc setBuffer:s_flag1      offset:0 atIndex:0];
    [enc setBuffer:s_flag2      offset:0 atIndex:1];
    [enc setBytes:&head_idx     length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs     length:sizeof(int) atIndex:3];
    [enc setBytes:&num_nbors    length:sizeof(int) atIndex:4];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * mtl_enforce_rules — clear flag1 if any nbor slot is 0 or > ngridmax.
 * ----------------------------------------------------------------------- */
extern "C" void mtl_enforce_rules(int head_idx, int num_octs, int ngridmax)
{
    if (num_octs <= 0) return;
    NSUInteger tg128 = 128;
    NSUInteger nblk  = ((NSUInteger)num_octs + tg128 - 1) / tg128;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_enforce_rules];
    [enc setBuffer:s_flag1    offset:0 atIndex:0];
    [enc setBuffer:s_nbor     offset:0 atIndex:1];
    [enc setBytes:&head_idx   length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs   length:sizeof(int) atIndex:3];
    [enc setBytes:&ngridmax   length:sizeof(int) atIndex:4];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg128,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* -----------------------------------------------------------------------
 * Batched flag helpers — one command buffer per operation, using
 * s_sort_fence for inter-encoder ordering and s_count_buf for the
 * atomic reduction result (zeroed via blit at the start of each cmd buf).
 *
 * mtl_init_flag_batch  : reset_flag1 [→ init_flag] → count_flag1  (1 sync)
 * mtl_user_flag_batch  : hydro_flag  → count_flag1                (1 sync)
 * mtl_smooth_flag_batch: 3×(count_neighbors → flag_count) → count (1 sync)
 * ----------------------------------------------------------------------- */

static int run_count_enc(id<MTLCommandBuffer> cmd,
                         int head_idx, int num_octs)
{
    NSUInteger tg1024 = 1024;
    NSUInteger total_cells = (NSUInteger)num_octs * 8;
    NSUInteger nblk  = (total_cells + tg1024 - 1) / tg1024;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc waitForFence:s_sort_fence];
    [enc setComputePipelineState:s_pso_count_flag1];
    [enc setBuffer:s_flag1    offset:0 atIndex:0];
    [enc setBuffer:s_count_buf offset:0 atIndex:1];
    [enc setBytes:&head_idx   length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs   length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg1024,1,1}];
    [enc endEncoding];   /* last encoder — no outgoing fence */
    return 0;            /* caller reads s_count_buf after waitUntilCompleted */
}

extern "C" int mtl_init_flag_batch(int head_coarse, int noct_coarse,
                                    int head_fine,   int noct_fine)
{
    if (noct_coarse <= 0) return 0;

    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];

    /* Zero the persistent count buffer */
    { id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      [blit fillBuffer:s_count_buf range:NSMakeRange(0, sizeof(int)) value:0];
      [blit updateFence:s_sort_fence];
      [blit endEncoding]; }

    /* reset_flag1 for coarse level */
    { NSUInteger nblk = ((NSUInteger)noct_coarse + 15) / 16;
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc waitForFence:s_sort_fence];
      [enc setComputePipelineState:s_pso_reset_flag1];
      [enc setBuffer:s_flag1    offset:0 atIndex:0];
      [enc setBytes:&head_coarse length:sizeof(int) atIndex:1];
      [enc setBytes:&noct_coarse length:sizeof(int) atIndex:2];
      [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
      [enc updateFence:s_sort_fence];
      [enc endEncoding]; }

    /* init_flag: propagate fine-level flags to coarse parent cells */
    if (noct_fine > 0) {
        NSUInteger nblk = ((NSUInteger)noct_fine + 127) / 128;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc waitForFence:s_sort_fence];
        [enc setComputePipelineState:s_pso_init_flag];
        [enc setBuffer:s_flag1    offset:0 atIndex:0];
        [enc setBuffer:s_grid     offset:0 atIndex:1];
        [enc setBuffer:s_father   offset:0 atIndex:2];
        [enc setBytes:&head_fine  length:sizeof(int) atIndex:3];
        [enc setBytes:&noct_fine  length:sizeof(int) atIndex:4];
        [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
        [enc updateFence:s_sort_fence];
        [enc endEncoding]; }

    run_count_enc(cmd, head_coarse, noct_coarse);
    [cmd commit]; [cmd waitUntilCompleted];
    return *(int *)s_count_buf.contents;
}

extern "C" int mtl_user_flag_batch(int head_idx, int num_octs,
                                    float gamma,      float smallr,  float smallc2,
                                    float err_grad_d, float err_grad_p,
                                    float floor_d,    float floor_p)
{
    if (num_octs <= 0) return 0;

    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];

    /* Zero count buffer */
    { id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      [blit fillBuffer:s_count_buf range:NSMakeRange(0, sizeof(int)) value:0];
      [blit updateFence:s_sort_fence];
      [blit endEncoding]; }

    /* hydro_flag */
    { NSUInteger nblk = ((NSUInteger)num_octs + 15) / 16;
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc waitForFence:s_sort_fence];
      [enc setComputePipelineState:s_pso_hydro_flag];
      [enc setBuffer:s_flag1      offset:0 atIndex:0];
      [enc setBuffer:s_nbor       offset:0 atIndex:1];
      [enc setBuffer:s_uold       offset:0 atIndex:2];
      [enc setBytes:&head_idx     length:sizeof(int)   atIndex:3];
      [enc setBytes:&num_octs     length:sizeof(int)   atIndex:4];
      [enc setBytes:&gamma        length:sizeof(float) atIndex:5];
      [enc setBytes:&smallr       length:sizeof(float) atIndex:6];
      [enc setBytes:&smallc2      length:sizeof(float) atIndex:7];
      [enc setBytes:&err_grad_d   length:sizeof(float) atIndex:8];
      [enc setBytes:&err_grad_p   length:sizeof(float) atIndex:9];
      [enc setBytes:&floor_d      length:sizeof(float) atIndex:10];
      [enc setBytes:&floor_p      length:sizeof(float) atIndex:11];
      [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
      [enc updateFence:s_sort_fence];
      [enc endEncoding]; }

    run_count_enc(cmd, head_idx, num_octs);
    [cmd commit]; [cmd waitUntilCompleted];
    return *(int *)s_count_buf.contents;
}

extern "C" int mtl_smooth_flag_batch(int head_idx, int num_octs)
{
    if (num_octs <= 0) return 0;

    /* n_nbor = {1, 2, 2} for NDIM=3 — mirrors metal_runner.f90 metal_smooth_flag */
    static const int n_nbor[3] = {1, 2, 2};
    NSUInteger nblk = ((NSUInteger)num_octs + 15) / 16;

    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];

    /* Zero count buffer */
    { id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      [blit fillBuffer:s_count_buf range:NSMakeRange(0, sizeof(int)) value:0];
      [blit updateFence:s_sort_fence];
      [blit endEncoding]; }

    /* 3 dilatation passes: count_neighbors → flag_count */
    for (int idim = 0; idim < 3; idim++) {
        int nn = n_nbor[idim];

        /* count_neighbors: flag2[cell,oct] = # flagged face-adjacent nbors */
        { id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
          [enc waitForFence:s_sort_fence];
          [enc setComputePipelineState:s_pso_count_neighbors];
          [enc setBuffer:s_flag2    offset:0 atIndex:0];
          [enc setBuffer:s_flag1    offset:0 atIndex:1];
          [enc setBuffer:s_nbor     offset:0 atIndex:2];
          [enc setBytes:&head_idx   length:sizeof(int) atIndex:3];
          [enc setBytes:&num_octs   length:sizeof(int) atIndex:4];
          [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
          [enc updateFence:s_sort_fence];
          [enc endEncoding]; }

        /* flag_count: promote flag1 if flag2 >= nn; clear flag2 if flag1 set */
        { id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
          [enc waitForFence:s_sort_fence];
          [enc setComputePipelineState:s_pso_flag_count];
          [enc setBuffer:s_flag1    offset:0 atIndex:0];
          [enc setBuffer:s_flag2    offset:0 atIndex:1];
          [enc setBytes:&head_idx   length:sizeof(int) atIndex:2];
          [enc setBytes:&num_octs   length:sizeof(int) atIndex:3];
          [enc setBytes:&nn         length:sizeof(int) atIndex:4];
          [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{128,1,1}];
          [enc updateFence:s_sort_fence];
          [enc endEncoding]; }
    }

    run_count_enc(cmd, head_idx, num_octs);
    [cmd commit]; [cmd waitUntilCompleted];
    return *(int *)s_count_buf.contents;
}

/* -----------------------------------------------------------------------
 * mtl_build_father — populate father[oct] = 1-based parent oct for each
 * oct at the given level.  Uses the hash table already populated by
 * mtl_insert_hash.  Needed before init_flag_kernel can run.
 * ckey_max_l and key_off_l are the per-level Hilbert parameters for
 * the PARENT level (ilevel - 1).
 * ----------------------------------------------------------------------- */
extern "C" void mtl_build_father(int head_idx, int num_octs,
                                  int hash_size,
                                  int ckey_max_l, long key_off_l)
{
    if (num_octs <= 0) return;
    NSUInteger tg128 = 128;
    NSUInteger nblk  = ((NSUInteger)num_octs + tg128 - 1) / tg128;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_update_father];
    [enc setBuffer:s_grid       offset:0 atIndex:0];
    [enc setBuffer:s_father     offset:0 atIndex:1];
    [enc setBuffer:s_hash_key   offset:0 atIndex:2];
    [enc setBuffer:s_hash_val   offset:0 atIndex:3];
    [enc setBytes:&hash_size    length:sizeof(int)  atIndex:4];
    [enc setBytes:&ckey_max_l   length:sizeof(int)  atIndex:5];
    [enc setBytes:&key_off_l    length:sizeof(long) atIndex:6];
    [enc setBytes:&head_idx     length:sizeof(int)  atIndex:7];
    [enc setBytes:&num_octs     length:sizeof(int)  atIndex:8];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg128,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* =======================================================================
 * AMR refine / sort / cache bridge functions
 * All mirrors of gpu_refine subroutines from gpu_runner.cuf.
 * ======================================================================= */

/* --- ifree_dev / ifree_cache_dev direct access (unified memory) -------- */

extern "C" void mtl_set_ifree(int val)
{
    ((int *)s_ifree_dev.contents)[0] = val;
}

extern "C" int mtl_get_ifree(void)
{
    return ((int *)s_ifree_dev.contents)[0];
}

extern "C" void mtl_set_ifree_cache(int val)
{
    ((int *)s_ifree_cache_dev.contents)[0] = val;
}

extern "C" int mtl_get_ifree_cache(void)
{
    return ((int *)s_ifree_cache_dev.contents)[0];
}

/* CPU-side increment — no kernel dispatch needed for unified memory. */
extern "C" void mtl_advance_ifree_cache(int new_noct)
{
    ((int *)s_ifree_cache_dev.contents)[0] += new_noct;
}

/* --- refine_kernel: create child octs for flagged cells ---------------- */

extern "C" void mtl_refine_cells(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, total = (NSUInteger)num_octs * 8;
    NSUInteger nblk = (total + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_refine];
    [enc setBuffer:s_grid      offset:0 atIndex:0];
    [enc setBuffer:s_flag1     offset:0 atIndex:1];
    [enc setBuffer:s_uold      offset:0 atIndex:2];
    [enc setBuffer:s_ifree_dev offset:0 atIndex:3];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:4];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:5];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- derefine_kernel: free child octs whose parent is no longer flagged */

extern "C" void mtl_derefine_cells(int head_idx, int num_octs, int hash_size)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_derefine];
    [enc setBuffer:s_grid         offset:0 atIndex:0];
    [enc setBuffer:s_flag1        offset:0 atIndex:1];
    [enc setBuffer:s_hash_key     offset:0 atIndex:2];
    [enc setBuffer:s_hash_val     offset:0 atIndex:3];
    [enc setBuffer:s_ckey_max_dev offset:0 atIndex:4];
    [enc setBuffer:s_key_off_dev  offset:0 atIndex:5];
    [enc setBytes:&hash_size length:sizeof(int) atIndex:6];
    [enc setBytes:&head_idx  length:sizeof(int) atIndex:7];
    [enc setBytes:&num_octs  length:sizeof(int) atIndex:8];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- free_hash_kernel: wipe hash entries for a range of octs ----------- */

extern "C" void mtl_free_hash_range(int head_idx, int num_octs, int hash_size)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_free_hash];
    [enc setBuffer:s_grid         offset:0 atIndex:0];
    [enc setBuffer:s_hash_key     offset:0 atIndex:1];
    [enc setBuffer:s_hash_val     offset:0 atIndex:2];
    [enc setBuffer:s_ckey_max_dev offset:0 atIndex:3];
    [enc setBuffer:s_key_off_dev  offset:0 atIndex:4];
    [enc setBytes:&hash_size length:sizeof(int) atIndex:5];
    [enc setBytes:&head_idx  length:sizeof(int) atIndex:6];
    [enc setBytes:&num_octs  length:sizeof(int) atIndex:7];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- update_hash_kernel: update hash entries after sort rearrangement --- */

extern "C" void mtl_update_hash_range(int head_idx, int num_octs, int hash_size)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_update_hash];
    [enc setBuffer:s_grid         offset:0 atIndex:0];
    [enc setBuffer:s_hash_key     offset:0 atIndex:1];
    [enc setBuffer:s_hash_val     offset:0 atIndex:2];
    [enc setBuffer:s_ckey_max_dev offset:0 atIndex:3];
    [enc setBuffer:s_key_off_dev  offset:0 atIndex:4];
    [enc setBytes:&hash_size length:sizeof(int) atIndex:5];
    [enc setBytes:&head_idx  length:sizeof(int) atIndex:6];
    [enc setBytes:&num_octs  length:sizeof(int) atIndex:7];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- insert_hash_all_kernel: insert newly created octs into hash ------- */

extern "C" void mtl_insert_hash_all(int head_idx, int num_octs, int hash_size)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_insert_hash_all];
    [enc setBuffer:s_grid         offset:0 atIndex:0];
    [enc setBuffer:s_hash_key     offset:0 atIndex:1];
    [enc setBuffer:s_hash_val     offset:0 atIndex:2];
    [enc setBuffer:s_ckey_max_dev offset:0 atIndex:3];
    [enc setBuffer:s_key_off_dev  offset:0 atIndex:4];
    [enc setBytes:&hash_size length:sizeof(int) atIndex:5];
    [enc setBytes:&head_idx  length:sizeof(int) atIndex:6];
    [enc setBytes:&num_octs  length:sizeof(int) atIndex:7];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- mtl_reset_hash: mirror gpu_reset_hash in gpu_runner.cuf ----------- *
 * Called once per coarse step (ilevel==levelmin).                         *
 * 1. Blit-zero s_hash_key / s_hash_val.                                   *
 * 2. Re-insert all real octs  [1 .. ifree-1].                             *
 * 3. Re-insert all cache octs [ngridmax+1 .. ngridmax+ifree_cache-1].    */
extern "C" void mtl_reset_hash(int ifree, int ngridmax, int ifree_cache,
                                int hash_size)
{
    /* Step 1: zero the hash table (GPU-side blit, ordered on the queue). */
    {
        id<MTLCommandBuffer>      cmd  = [s_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit fillBuffer:s_hash_key range:NSMakeRange(0, (NSUInteger)hash_size * sizeof(long)) value:0];
        [blit fillBuffer:s_hash_val range:NSMakeRange(0, (NSUInteger)hash_size * sizeof(int))  value:0];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    /* Step 2: insert real octs [1 .. ifree-1]. */
    {
        int num = ifree - 1;
        if (num > 0) {
            NSUInteger tg = 128, nblk = ((NSUInteger)num + tg - 1) / tg;
            id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:s_pso_insert_hash_all];
            [enc setBuffer:s_grid         offset:0 atIndex:0];
            [enc setBuffer:s_hash_key     offset:0 atIndex:1];
            [enc setBuffer:s_hash_val     offset:0 atIndex:2];
            [enc setBuffer:s_ckey_max_dev offset:0 atIndex:3];
            [enc setBuffer:s_key_off_dev  offset:0 atIndex:4];
            [enc setBytes:&hash_size length:sizeof(int) atIndex:5];
            int head = 1;
            [enc setBytes:&head      length:sizeof(int) atIndex:6];
            [enc setBytes:&num       length:sizeof(int) atIndex:7];
            [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    /* Step 3: insert cache octs [ngridmax+1 .. ngridmax+ifree_cache-1]. */
    {
        int cache_head = ngridmax + 1;
        int cache_num  = ifree_cache - 1;
        if (cache_num > 0) {
            NSUInteger tg = 128, nblk = ((NSUInteger)cache_num + tg - 1) / tg;
            id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:s_pso_insert_hash_all];
            [enc setBuffer:s_grid         offset:0 atIndex:0];
            [enc setBuffer:s_hash_key     offset:0 atIndex:1];
            [enc setBuffer:s_hash_val     offset:0 atIndex:2];
            [enc setBuffer:s_ckey_max_dev offset:0 atIndex:3];
            [enc setBuffer:s_key_off_dev  offset:0 atIndex:4];
            [enc setBytes:&hash_size   length:sizeof(int) atIndex:5];
            [enc setBytes:&cache_head  length:sizeof(int) atIndex:6];
            [enc setBytes:&cache_num   length:sizeof(int) atIndex:7];
            [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
}

/* --- init_global_swap_table_kernel: identity permutation --------------- */

extern "C" void mtl_init_swap_table(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_init_swap_table];
    [enc setBuffer:s_swap_global offset:0 atIndex:0];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:1];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:2];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- init_prefix_sum_level_kernel: bit = (lev != ilevel) -------------- */

extern "C" void mtl_init_prefix_level(int head_idx, int num_octs, int ilevel)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_prefix_level];
    [enc setBuffer:s_grid        offset:0 atIndex:0];
    [enc setBuffer:s_swap_global offset:0 atIndex:1];
    [enc setBuffer:s_prefix_sum  offset:0 atIndex:2];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:3];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:4];
    [enc setBytes:&ilevel   length:sizeof(int) atIndex:5];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- init_prefix_sum_bit_kernel: bit ibit of Hilbert key -------------- */

extern "C" void mtl_init_prefix_bit(int head_idx, int num_octs, int ibit)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_prefix_bit];
    [enc setBuffer:s_grid        offset:0 atIndex:0];
    [enc setBuffer:s_swap_global offset:0 atIndex:1];
    [enc setBuffer:s_prefix_sum  offset:0 atIndex:2];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:3];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:4];
    [enc setBytes:&ibit     length:sizeof(int) atIndex:5];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- compute_local_swap_table_kernel: LSD scatter ---------------------- */

extern "C" void mtl_compute_local_swap(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_local_swap];
    [enc setBuffer:s_swap_local  offset:0 atIndex:0];
    [enc setBuffer:s_swap_global offset:0 atIndex:1];
    [enc setBuffer:s_prefix_sum  offset:0 atIndex:2];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:3];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:4];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- update_global_swap_table_kernel: apply local swap to global ------- */

extern "C" void mtl_update_global_swap(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_global_swap];
    [enc setBuffer:s_swap_global offset:0 atIndex:0];
    [enc setBuffer:s_swap_local  offset:0 atIndex:1];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- scan_phase_fenced: one scan pass in a new encoder with fence I/O ---
 * Waits on fence_in before dispatching, updates fence_out after.
 * fence_in and fence_out may be the same object (sequential chaining).   */
static void scan_phase_fenced(id<MTLComputePipelineState> pso,
                              id<MTLBuffer> data, id<MTLBuffer> psums,
                              int offset, int n,
                              id<MTLCommandBuffer> cmd,
                              id<MTLFence> fence_in, id<MTLFence> fence_out)
{
    NSUInteger tg = 256, nblk = ((NSUInteger)n + tg - 1) / tg;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc waitForFence:fence_in];
    [enc setComputePipelineState:pso];
    [enc setBuffer:data  offset:0 atIndex:0];
    [enc setBuffer:psums offset:0 atIndex:1];
    [enc setBytes:&offset length:sizeof(int) atIndex:2];
    [enc setBytes:&n      length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc updateFence:fence_out];
    [enc endEncoding];
}

/* --- mtl_hilbert_sort_level: one command buffer per bit pass -----------
 * Each pass contains 5–7 compute encoders connected by s_sort_fence.
 * Reduces commit/waitUntilCompleted from 6*num_bits to num_bits — a 6×
 * reduction — while guaranteeing memory visibility via Metal fences.     */
extern "C" void mtl_hilbert_sort_level(int head_idx, int num_octs, int num_bits)
{
    if (num_octs <= 0 || num_bits <= 0) return;

    int offset    = head_idx - 1;
    int n         = num_octs;
    int nblk_sort = (n + 127) / 128;
    int nblk_scan = (n + 255) / 256;
    bool need_fixup = (nblk_scan > 1);

    for (int ibit = 0; ibit < num_bits; ibit++) {
        id<MTLCommandBuffer> cmd = [s_queue commandBuffer];

        /* 1. init_prefix_bit — no fence_in (first encoder in this buffer) */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:s_pso_prefix_bit];
            [enc setBuffer:s_grid        offset:0 atIndex:0];
            [enc setBuffer:s_swap_global offset:0 atIndex:1];
            [enc setBuffer:s_prefix_sum  offset:0 atIndex:2];
            [enc setBytes:&head_idx length:sizeof(int) atIndex:3];
            [enc setBytes:&num_octs length:sizeof(int) atIndex:4];
            [enc setBytes:&ibit     length:sizeof(int) atIndex:5];
            [enc dispatchThreadgroups:{(NSUInteger)nblk_sort,1,1}
                 threadsPerThreadgroup:{128,1,1}];
            [enc updateFence:s_sort_fence];
            [enc endEncoding];
        }
        /* 2. prefix scan: phase 1 */
        scan_phase_fenced(s_pso_scan_block, s_prefix_sum, s_partial_sums,
                          offset, n, cmd, s_sort_fence, s_sort_fence);
        if (need_fixup) {
            int nblk_scan2 = (nblk_scan + 255) / 256;
            /* phase 2: scan partial_sums; block totals → partial_sums_2 (separate buffer) */
            scan_phase_fenced(s_pso_scan_block, s_partial_sums, s_partial_sums_2,
                              0, nblk_scan, cmd, s_sort_fence, s_sort_fence);
            if (nblk_scan2 > 1) {
                /* nblk_scan > 256: need a third level.
                 * single-block scan of partial_sums_2; block totals → partial_sums_3. */
                scan_phase_fenced(s_pso_scan_block, s_partial_sums_2, s_partial_sums_3,
                                  0, nblk_scan2, cmd, s_sort_fence, s_sort_fence);
                /* fixup partial_sums using partial_sums_2 */
                scan_phase_fenced(s_pso_scan_fixup, s_partial_sums, s_partial_sums_2,
                                  0, nblk_scan, cmd, s_sort_fence, s_sort_fence);
            }
            /* phase 3: fixup prefix_sum using partial_sums */
            scan_phase_fenced(s_pso_scan_fixup, s_prefix_sum, s_partial_sums,
                              offset, n, cmd, s_sort_fence, s_sort_fence);
        }
        /* 3. compute_local_swap */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc waitForFence:s_sort_fence];
            [enc setComputePipelineState:s_pso_local_swap];
            [enc setBuffer:s_swap_local  offset:0 atIndex:0];
            [enc setBuffer:s_swap_global offset:0 atIndex:1];
            [enc setBuffer:s_prefix_sum  offset:0 atIndex:2];
            [enc setBytes:&head_idx length:sizeof(int) atIndex:3];
            [enc setBytes:&num_octs length:sizeof(int) atIndex:4];
            [enc dispatchThreadgroups:{(NSUInteger)nblk_sort,1,1}
                 threadsPerThreadgroup:{128,1,1}];
            [enc updateFence:s_sort_fence];
            [enc endEncoding];
        }
        /* 4. update_global_swap — last encoder, no outgoing fence needed */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc waitForFence:s_sort_fence];
            [enc setComputePipelineState:s_pso_global_swap];
            [enc setBuffer:s_swap_global offset:0 atIndex:0];
            [enc setBuffer:s_swap_local  offset:0 atIndex:1];
            [enc setBytes:&head_idx length:sizeof(int) atIndex:2];
            [enc setBytes:&num_octs length:sizeof(int) atIndex:3];
            [enc dispatchThreadgroups:{(NSUInteger)nblk_sort,1,1}
                 threadsPerThreadgroup:{128,1,1}];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

/* --- sort_gather_grid_kernel: pack grid metadata into flag2 scratch ---- */

extern "C" void mtl_sort_gather_grid(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_gather_grid];
    [enc setBuffer:s_flag2       offset:0 atIndex:0];
    [enc setBuffer:s_grid        offset:0 atIndex:1];
    [enc setBuffer:s_swap_global offset:0 atIndex:2];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:3];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:4];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- sort_scatter_grid_kernel: unpack flag2 → grid, recompute hkey ---- */

extern "C" void mtl_sort_scatter_grid(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_scatter_grid];
    [enc setBuffer:s_grid  offset:0 atIndex:0];
    [enc setBuffer:s_flag2 offset:0 atIndex:1];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- sort_gather_flag_kernel: flag2[:,oct] = flag1[:,swap_global[oct]] - */

extern "C" void mtl_sort_gather_flag(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_gather_flag];
    [enc setBuffer:s_flag2       offset:0 atIndex:0];
    [enc setBuffer:s_flag1       offset:0 atIndex:1];
    [enc setBuffer:s_swap_global offset:0 atIndex:2];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:3];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:4];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- sort_scatter_flag_kernel: flag1[:,oct] = flag2[:,oct] ------------- */

extern "C" void mtl_sort_scatter_flag(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_octs + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_scatter_flag];
    [enc setBuffer:s_flag1 offset:0 atIndex:0];
    [enc setBuffer:s_flag2 offset:0 atIndex:1];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:2];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- sort_gather_hydro_kernel: unew[:,oct] = uold[:,swap_global[oct]] -- */

extern "C" void mtl_sort_gather_hydro(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger tg = 128, total = (NSUInteger)num_octs * 8;
    NSUInteger nblk = (total + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_gather_hydro];
    [enc setBuffer:s_unew        offset:0 atIndex:0];
    [enc setBuffer:s_uold        offset:0 atIndex:1];
    [enc setBuffer:s_swap_global offset:0 atIndex:2];
    [enc setBytes:&head_idx length:sizeof(int) atIndex:3];
    [enc setBytes:&num_octs length:sizeof(int) atIndex:4];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- mtl_blit_unew_to_uold: copy unew[head..] → uold (mirrors cudaMemcpy) */

extern "C" void mtl_blit_unew_to_uold(int head_idx, int num_octs)
{
    if (num_octs <= 0) return;
    NSUInteger oct_floats = (NSUInteger)s_nvar * 8;
    NSUInteger src_off    = (NSUInteger)(head_idx - 1) * oct_floats * sizeof(float);
    NSUInteger blit_len   = (NSUInteger)num_octs * oct_floats * sizeof(float);
    id<MTLCommandBuffer>      cmd  = [s_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:s_unew sourceOffset:src_off
               toBuffer:s_uold destinationOffset:src_off size:blit_len];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- update_nbor_prefix_kernel: build one nbor column + prefix_sum ----- */

extern "C" void mtl_update_nbor_prefix(int head_idx, int num_subgrids,
                                       int hash_size, int input_ind)
{
    if (num_subgrids <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_subgrids + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_nbor_prefix];
    [enc setBuffer:s_nbor             offset:0 atIndex:0];
    [enc setBuffer:s_grid             offset:0 atIndex:1];
    [enc setBuffer:s_hash_key         offset:0 atIndex:2];
    [enc setBuffer:s_hash_val         offset:0 atIndex:3];
    [enc setBuffer:s_ckey_max_dev     offset:0 atIndex:4];
    [enc setBuffer:s_key_off_dev      offset:0 atIndex:5];
    [enc setBuffer:s_box_ckey_min_dev offset:0 atIndex:6];
    [enc setBuffer:s_box_ckey_max_dev offset:0 atIndex:7];
    [enc setBuffer:s_periodic_dev     offset:0 atIndex:8];
    [enc setBuffer:s_prefix_sum       offset:0 atIndex:9];
    [enc setBytes:&hash_size     length:sizeof(int) atIndex:10];
    [enc setBytes:&head_idx      length:sizeof(int) atIndex:11];
    [enc setBytes:&num_subgrids  length:sizeof(int) atIndex:12];
    [enc setBytes:&input_ind     length:sizeof(int) atIndex:13];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- compute_cache_swap_table_kernel: select subgrids missing a neighbour */

extern "C" void mtl_compute_cache_swap(int head_idx, int num_subgrids)
{
    if (num_subgrids <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)num_subgrids + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_cache_swap];
    [enc setBuffer:s_swap_local  offset:0 atIndex:0];
    [enc setBuffer:s_prefix_sum  offset:0 atIndex:1];
    [enc setBytes:&head_idx     length:sizeof(int) atIndex:2];
    [enc setBytes:&num_subgrids length:sizeof(int) atIndex:3];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- make_cache_octs_kernel: create ghost octs in the cache region ----- */

extern "C" void mtl_make_cache_octs(int head_idx, int num_subgrids,
                                     int hash_size, int ngridmax,
                                     int ifree_cache, int new_noct,
                                     int input_ind)
{
    if (new_noct <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)new_noct + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_make_cache];
    [enc setBuffer:s_grid             offset:0 atIndex:0];
    [enc setBuffer:s_flag1            offset:0 atIndex:1];
    [enc setBuffer:s_uold             offset:0 atIndex:2];
    [enc setBuffer:s_swap_local       offset:0 atIndex:3];
    [enc setBuffer:s_father           offset:0 atIndex:4];
    [enc setBuffer:s_nbor             offset:0 atIndex:5];
    [enc setBuffer:s_hash_key         offset:0 atIndex:6];
    [enc setBuffer:s_hash_val         offset:0 atIndex:7];
    [enc setBuffer:s_ckey_max_dev     offset:0 atIndex:8];
    [enc setBuffer:s_key_off_dev      offset:0 atIndex:9];
    [enc setBuffer:s_box_ckey_min_dev offset:0 atIndex:10];
    [enc setBuffer:s_box_ckey_max_dev offset:0 atIndex:11];
    [enc setBuffer:s_periodic_dev     offset:0 atIndex:12];
    [enc setBytes:&hash_size   length:sizeof(int) atIndex:13];
    [enc setBytes:&ngridmax    length:sizeof(int) atIndex:14];
    [enc setBytes:&ifree_cache length:sizeof(int) atIndex:15];
    [enc setBytes:&new_noct    length:sizeof(int) atIndex:16];
    [enc setBytes:&input_ind   length:sizeof(int) atIndex:17];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- mtl_nbor_scan: update_nbor_prefix + all scan phases in one cmd buf -
 * Replaces separate mtl_update_nbor_prefix + mtl_prefix_scan + mtl_get_prefix_total.
 * Returns the inclusive-scan total (= number of subgrids missing neighbour ind).
 * Uses s_sort_fence for inter-encoder ordering within the command buffer.  */
extern "C" int mtl_nbor_scan(int head_idx, int num_subgrids,
                              int hash_size, int input_ind)
{
    if (num_subgrids <= 0) return 0;

    int offset    = head_idx - 1;   /* 0-based for scan */
    int n         = num_subgrids;
    int nblk_nbor = (n + 127) / 128;
    int nblk_scan = (n + 255) / 256;
    bool need_fixup = (nblk_scan > 1);

    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];

    /* 1. update_nbor_prefix_kernel — first encoder, no waitForFence */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s_pso_nbor_prefix];
        [enc setBuffer:s_nbor             offset:0 atIndex:0];
        [enc setBuffer:s_grid             offset:0 atIndex:1];
        [enc setBuffer:s_hash_key         offset:0 atIndex:2];
        [enc setBuffer:s_hash_val         offset:0 atIndex:3];
        [enc setBuffer:s_ckey_max_dev     offset:0 atIndex:4];
        [enc setBuffer:s_key_off_dev      offset:0 atIndex:5];
        [enc setBuffer:s_box_ckey_min_dev offset:0 atIndex:6];
        [enc setBuffer:s_box_ckey_max_dev offset:0 atIndex:7];
        [enc setBuffer:s_periodic_dev     offset:0 atIndex:8];
        [enc setBuffer:s_prefix_sum       offset:0 atIndex:9];
        [enc setBytes:&hash_size    length:sizeof(int) atIndex:10];
        [enc setBytes:&head_idx     length:sizeof(int) atIndex:11];
        [enc setBytes:&num_subgrids length:sizeof(int) atIndex:12];
        [enc setBytes:&input_ind    length:sizeof(int) atIndex:13];
        [enc dispatchThreadgroups:{(NSUInteger)nblk_nbor,1,1}
             threadsPerThreadgroup:{128,1,1}];
        [enc updateFence:s_sort_fence];
        [enc endEncoding];
    }
    /* 2. prefix scan: phase 1 */
    scan_phase_fenced(s_pso_scan_block, s_prefix_sum, s_partial_sums,
                      offset, n, cmd, s_sort_fence, s_sort_fence);
    if (need_fixup) {
        int nblk_scan2 = (nblk_scan + 255) / 256;
        scan_phase_fenced(s_pso_scan_block, s_partial_sums, s_partial_sums_2,
                          0, nblk_scan, cmd, s_sort_fence, s_sort_fence);
        if (nblk_scan2 > 1) {
            scan_phase_fenced(s_pso_scan_block, s_partial_sums_2, s_partial_sums_3,
                              0, nblk_scan2, cmd, s_sort_fence, s_sort_fence);
            scan_phase_fenced(s_pso_scan_fixup, s_partial_sums, s_partial_sums_2,
                              0, nblk_scan, cmd, s_sort_fence, s_sort_fence);
        }
        scan_phase_fenced(s_pso_scan_fixup, s_prefix_sum, s_partial_sums,
                          offset, n, cmd, s_sort_fence, s_sort_fence);
    }

    [cmd commit];
    [cmd waitUntilCompleted];

    /* Read total from shared memory — GPU done, no sync needed. */
    return ((int *)s_prefix_sum.contents)[offset + n - 1];
}

/* --- mtl_cache_fill: compute_cache_swap + make_cache_octs + insert_hash -
 * Replaces three separate commit/wait calls with one.
 * Uses s_sort_fence for inter-encoder ordering.                           */
extern "C" void mtl_cache_fill(int head_idx, int num_subgrids,
                                int hash_size, int input_ind,
                                int ngridmax, int ifree_cache, int new_noct)
{
    if (new_noct <= 0) return;

    int nblk_sg    = (num_subgrids + 127) / 128;
    int nblk_cache = (new_noct     + 127) / 128;

    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];

    /* 1. compute_cache_swap_table_kernel */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s_pso_cache_swap];
        [enc setBuffer:s_swap_local  offset:0 atIndex:0];
        [enc setBuffer:s_prefix_sum  offset:0 atIndex:1];
        [enc setBytes:&head_idx     length:sizeof(int) atIndex:2];
        [enc setBytes:&num_subgrids length:sizeof(int) atIndex:3];
        [enc dispatchThreadgroups:{(NSUInteger)nblk_sg,1,1}
             threadsPerThreadgroup:{128,1,1}];
        [enc updateFence:s_sort_fence];
        [enc endEncoding];
    }
    /* 2. make_cache_octs_kernel */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc waitForFence:s_sort_fence];
        [enc setComputePipelineState:s_pso_make_cache];
        [enc setBuffer:s_grid             offset:0 atIndex:0];
        [enc setBuffer:s_flag1            offset:0 atIndex:1];
        [enc setBuffer:s_uold             offset:0 atIndex:2];
        [enc setBuffer:s_swap_local       offset:0 atIndex:3];
        [enc setBuffer:s_father           offset:0 atIndex:4];
        [enc setBuffer:s_nbor             offset:0 atIndex:5];
        [enc setBuffer:s_hash_key         offset:0 atIndex:6];
        [enc setBuffer:s_hash_val         offset:0 atIndex:7];
        [enc setBuffer:s_ckey_max_dev     offset:0 atIndex:8];
        [enc setBuffer:s_key_off_dev      offset:0 atIndex:9];
        [enc setBuffer:s_box_ckey_min_dev offset:0 atIndex:10];
        [enc setBuffer:s_box_ckey_max_dev offset:0 atIndex:11];
        [enc setBuffer:s_periodic_dev     offset:0 atIndex:12];
        [enc setBytes:&hash_size    length:sizeof(int) atIndex:13];
        [enc setBytes:&ngridmax     length:sizeof(int) atIndex:14];
        [enc setBytes:&ifree_cache  length:sizeof(int) atIndex:15];
        [enc setBytes:&new_noct     length:sizeof(int) atIndex:16];
        [enc setBytes:&input_ind    length:sizeof(int) atIndex:17];
        [enc dispatchThreadgroups:{(NSUInteger)nblk_cache,1,1}
             threadsPerThreadgroup:{128,1,1}];
        [enc updateFence:s_sort_fence];
        [enc endEncoding];
    }
    /* 3. insert_hash_cache_kernel */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc waitForFence:s_sort_fence];
        [enc setComputePipelineState:s_pso_insert_hash_cache];
        [enc setBuffer:s_grid         offset:0 atIndex:0];
        [enc setBuffer:s_hash_key     offset:0 atIndex:1];
        [enc setBuffer:s_hash_val     offset:0 atIndex:2];
        [enc setBuffer:s_ckey_max_dev offset:0 atIndex:3];
        [enc setBuffer:s_key_off_dev  offset:0 atIndex:4];
        [enc setBytes:&hash_size    length:sizeof(int) atIndex:5];
        [enc setBytes:&ngridmax     length:sizeof(int) atIndex:6];
        [enc setBytes:&ifree_cache  length:sizeof(int) atIndex:7];
        [enc setBytes:&new_noct     length:sizeof(int) atIndex:8];
        [enc dispatchThreadgroups:{(NSUInteger)nblk_cache,1,1}
             threadsPerThreadgroup:{128,1,1}];
        [enc endEncoding];   /* last encoder — no outgoing fence needed */
    }

    [cmd commit];
    [cmd waitUntilCompleted];
}

/* --- insert_hash_cache_kernel: insert new cache octs into hash table --- */

extern "C" void mtl_insert_hash_cache_r(int hash_size, int ngridmax,
                                         int ifree_cache, int new_noct)
{
    if (new_noct <= 0) return;
    NSUInteger tg = 128, nblk = ((NSUInteger)new_noct + tg - 1) / tg;
    id<MTLCommandBuffer>         cmd = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s_pso_insert_hash_cache];
    [enc setBuffer:s_grid         offset:0 atIndex:0];
    [enc setBuffer:s_hash_key     offset:0 atIndex:1];
    [enc setBuffer:s_hash_val     offset:0 atIndex:2];
    [enc setBuffer:s_ckey_max_dev offset:0 atIndex:3];
    [enc setBuffer:s_key_off_dev  offset:0 atIndex:4];
    [enc setBytes:&hash_size   length:sizeof(int) atIndex:5];
    [enc setBytes:&ngridmax    length:sizeof(int) atIndex:6];
    [enc setBytes:&ifree_cache length:sizeof(int) atIndex:7];
    [enc setBytes:&new_noct    length:sizeof(int) atIndex:8];
    [enc dispatchThreadgroups:{nblk,1,1} threadsPerThreadgroup:{tg,1,1}];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}
