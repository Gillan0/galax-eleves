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

    const b_type MAGIC_TEN = b_type(10.0f);
    const b_type MAGIC_ONE = b_type(1.0f);
    const b_type MAGIC_ZERO = b_type(0.0f);

    const b_type MAGIC_DT_VEL = b_type(2.0f);
    const b_type MAGIC_DT_POS = b_type(0.1f);

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i+= b_type::size)
    {
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        
        b_type raccx_i = MAGIC_ZERO;
        b_type raccy_i = MAGIC_ZERO;
        b_type raccz_i = MAGIC_ZERO;


        for (int j = 0; j < n_particles; j+= b_type::size)
        {                
            b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
            b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
            b_type rposz_j = b_type::load_unaligned(&particles.z[j]);
            b_type rmass_j = b_type::load_unaligned(&initstate.masses[j]);
            
            for (int k = 0; k < b_type::size; k++) 
            {

                const b_type diffx = xsimd::sub(rposx_j, rposx_i);
                const b_type diffy = xsimd::sub(rposy_j, rposy_i);
                const b_type diffz = xsimd::sub(rposz_j, rposz_i);

                b_type dij = xsimd::fma(diffx, diffx, MAGIC_ZERO);
                dij = xsimd::fma(diffy, diffy, dij);
                dij = xsimd::fma(diffz, diffz, dij);
                
                /* Calcul des distances */
                const auto if_greater_than_1 = xsimd::gt(dij, MAGIC_ONE);
                
                dij = xsimd::rsqrt(dij);
                dij = MAGIC_TEN * dij * dij * dij;
                dij = xsimd::select(if_greater_than_1, dij, MAGIC_ONE) * rmass_j;

                raccx_i = xsimd::fma(diffx, dij, raccx_i);
                raccy_i = xsimd::fma(diffy, dij, raccy_i);
                raccz_i = xsimd::fma(diffz, dij, raccz_i);

                rposx_j = xs::rotate_right<1>(rposx_j);
                rposy_j = xs::rotate_right<1>(rposy_j);
                rposz_j = xs::rotate_right<1>(rposz_j);
                rmass_j = xs::rotate_right<1>(rmass_j);

            }
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
        
        const b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        const b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        const b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        rvelx_i = xsimd::fma(raccx_i, MAGIC_DT_VEL, rvelx_i);
        rvely_i = xsimd::fma(raccy_i, MAGIC_DT_VEL, rvely_i);
        rvelz_i = xsimd::fma(raccz_i, MAGIC_DT_VEL, rvelz_i);

        rposx_i = xsimd::fma(rvelx_i, MAGIC_DT_POS, rposx_i);
        rposy_i = xsimd::fma(rvely_i, MAGIC_DT_POS, rposy_i);
        rposz_i = xsimd::fma(rvelz_i, MAGIC_DT_POS, rposz_i);

        rvelx_i.store_unaligned(&velocitiesx[i]);
        rvely_i.store_unaligned(&velocitiesy[i]);
        rvelz_i.store_unaligned(&velocitiesz[i]);

        rposx_i.store_unaligned(&particles.x[i]);
        rposy_i.store_unaligned(&particles.y[i]);
        rposz_i.store_unaligned(&particles.z[i]);

    }
}

#endif // GALAX_MODEL_CPU_FAST
