#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
#define BLOCK_SIZE (128)

__global__ void compute_acc(float4 * positionsGPU, float3 * accelerationsGPU, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    float4 pi = positionsGPU[i];

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    __shared__ float4 tile[BLOCK_SIZE];

    // Loop over tiles
    for (int tileStart = 0; tileStart < n_particles; tileStart += BLOCK_SIZE)
    {
        int j = tileStart + threadIdx.x;

        if (j < n_particles)
            tile[threadIdx.x] = positionsGPU[j];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            int idx = tileStart + k;
            if (idx >= n_particles) break;

            float4 pj = tile[k];

            float diffx = pj.x - pi.x;
            float diffy = pj.y - pi.y;
            float diffz = pj.z - pi.z;

            float dij = fmaf(diffx, diffx, 0.0f);
			dij = fmaf(diffy, diffy, dij);
			dij = fmaf(diffz, diffz, dij);

			dij = rsqrtf(dij);
			dij = fminf(10.0f, 10.0f * dij * dij * dij);
			dij = dij *  pj.w;

			ax = fmaf(diffx, dij, ax);
			ay = fmaf(diffz, dij, ay);
			az = fmaf(diffy, dij, az);
        }

        __syncthreads();
    }

    accelerationsGPU[i].x = ax;
    accelerationsGPU[i].y = ay;
    accelerationsGPU[i].z = az;
}


__global__ void maj_pos(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles) return;

	velocitiesGPU[i].x =  fmaf(accelerationsGPU[i].x, 2.0f, velocitiesGPU[i].x);
	velocitiesGPU[i].y =  fmaf(accelerationsGPU[i].y, 2.0f, velocitiesGPU[i].y);
	velocitiesGPU[i].z =  fmaf(accelerationsGPU[i].z, 2.0f, velocitiesGPU[i].z);
	positionsGPU[i].x = fmaf(velocitiesGPU[i].x, DIFF_T, positionsGPU[i].x);
	positionsGPU[i].y = fmaf(velocitiesGPU[i].y, DIFF_T, positionsGPU[i].y);
	positionsGPU[i].z = fmaf(velocitiesGPU[i].z, DIFF_T, positionsGPU[i].z);

}

void update_position_cu(float4* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
	int nthreads = BLOCK_SIZE;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
