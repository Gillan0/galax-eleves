#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i+= b_type::size)
	{
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

		for (int j = 0; j < n_particles; j++)
		{
            b_type rposx_j = b_type(particles.x[j]);
            b_type rposy_j = b_type(particles.y[j]);
            b_type rposz_j = b_type(particles.z[j]);
            b_type rmass_j = b_type(initstate.masses[j]);
			
            b_type diffx = rposx_j - rposx_i;
            b_type diffy = rposy_j - rposy_i;
            b_type diffz = rposz_j - rposz_i;

            b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;
            
            auto is_self = xsimd::eq(dij, b_type(0.0f));

            b_type inv_r = xsimd::rsqrt(dij);

            b_type inv_r3 = inv_r * inv_r * inv_r;
            
            b_type final_d = b_type(10.0f) * inv_r3;

            auto if_greater_than_1 = xsimd::gt(dij, b_type(1.0));
            
            final_d = xsimd::select(if_greater_than_1, final_d, b_type(10.0));
            
            final_d = xsimd::select(is_self, b_type(0.0f), final_d);

            raccx_i += diffx * final_d * rmass_j;
            raccy_i += diffy * final_d * rmass_j;
            raccz_i += diffz * final_d * rmass_j;
        }


        raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);

    }
	
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {            

        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);

        b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
        b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
        b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);
        
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        rvelx_i += raccx_i * 2.0f;
        rvely_i += raccy_i * 2.0f;
        rvelz_i += raccz_i * 2.0f;

        rposx_i += rvelx_i * 0.1f;
        rposy_i += rvely_i * 0.1f;
        rposz_i += rvelz_i * 0.1f;

        rvelx_i.store_unaligned(&velocitiesx[i]);
        rvely_i.store_unaligned(&velocitiesy[i]);
        rvelz_i.store_unaligned(&velocitiesz[i]);

        rposx_i.store_unaligned(&particles.x[i]);
        rposy_i.store_unaligned(&particles.y[i]);
        rposz_i.store_unaligned(&particles.z[i]);

    }
}


// OMP  version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i ++)
//     {
//     }


// OMP + xsimd version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i += b_type::size)
//     {
//         // load registers body i
//         const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
//         const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
//         const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
//               b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
//               b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
//               b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

//         ...
//     }

    
/*
    #pragma omp parallel for 
        for (int i = 0; i < n_particles; i ++)
        {
            for (int j = 0; j < n_particles; j++)
            {
                if(i != j)
                {
                    // align vector ?
                    const float diffx = particles.x[j] - particles.x[i];
                    const float diffy = particles.y[j] - particles.y[i];
                    const float diffz = particles.z[j] - particles.z[i];

                    float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                    // remove the if case if possible
                    if (dij < 1.0)
                    {
                        dij = 10.0;
                    }
                    else
                    {
                        dij = std::sqrt(dij); // Fast sqrt algo ?

                        dij = 10.0 / (dij * dij * dij);
                    }

                    accelerationsx[i] += diffx * dij * initstate.masses[j];
                    accelerationsy[i] += diffy * dij * initstate.masses[j];
                    accelerationsz[i] += diffz * dij * initstate.masses[j];
                }
            }
        
        }
    
    
    #pragma omp parallel for 
    for (int i = 0; i < n_particles; i++)
    {            
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;

    }
*/    
    // Options de vectorialisation à la compilation
    // FAST MATH

/*
    #pragma omp parallel for 
    for (int i = 0; i < n_particles; i ++)
    {
        for (int j = 0; j < n_particles; j++)
        {
            // Distance 2x 
            const float diffx = particles.x[j] - particles.x[i];
            const float diffy = particles.y[j] - particles.y[i];
            const float diffz = particles.z[j] - particles.z[i];

            float dij = diffx * diffx + diffy * diffy + diffz * diffz;

            dij = std::sqrt(dij); // Appel de la réciproque de la racine
            // Check sur option de compilation

            dij = std::max(10.0 / (dij * dij * dij), 10.0); 
            // Mask / operation booleéennes vectorielles

            accelerationsx[i] += diffx * dij * initstate.masses[j];
            accelerationsy[i] += diffy * dij * initstate.masses[j];
            accelerationsz[i] += diffz * dij * initstate.masses[j];
        }
        
    }
        
    // SIMD ici ou GPU : 
    #pragma omp parallel for 
    for (int i = 0; i < n_particles; i++)
    {            
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;

    }

}
*/

    // OMP + xsimd version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i += b_type::size)
//     {
//         // load registers body i
//         const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
//         const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
//         const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
//               b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
//               b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
//               b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

//         ...
//     }

#endif // GALAX_MODEL_CPU_FAST
