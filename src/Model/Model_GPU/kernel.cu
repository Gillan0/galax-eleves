#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

	float posX = positionsGPU[i].x;
	float posY = positionsGPU[i].y;
	float posZ = positionsGPU[i].z;

    for (int j = 0; j < n_particles; j++)
    {
        if (i == j) continue;

		const float diffx = positionsGPU[j].x - posX;
		const float diffy = positionsGPU[j].y - posY;
		const float diffz = positionsGPU[j].z - posZ;

		float dij = diffx * diffx + diffy * diffy + diffz * diffz;

		dij = rsqrtf(dij);
		dij = fminf(10.0f, 10.0f * dij * dij * dij);

        ax += diffx * dij * massesGPU[j];
        ay += diffy * dij * massesGPU[j];
        az += diffz * dij * massesGPU[j];
    }

    accelerationsGPU[i].x += ax;
    accelerationsGPU[i].y += ay;
    accelerationsGPU[i].z += az;
}


__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
	positionsGPU[i].x += velocitiesGPU[i].x * DIFF_T;
	positionsGPU[i].y += velocitiesGPU[i].y * DIFF_T;
	positionsGPU[i].z += velocitiesGPU[i].z * DIFF_T;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
