#ifndef math__rng_hxx
#define math__rng_hxx

/**
 *  \brief  Probability (various functions)
 *
 *  \date    2015/04/19
 *  \author  Vaclav Krpec  <vencik@razdva.cz>
 *
 *
 *  LEGAL NOTICE
 *
 *  Copyright (c) 2015, Vaclav Krpec
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 *  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "config.hxx"
#include "real.hxx"

#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cassert>
#include <cmath>

#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


/** Default probability precision factor (4 f.p. digits) */
#define MATH_DEFAULT_RAND_SCALE 10000


namespace math {

/**
 *  \brief  Random floating point number within range
 *
 *  \param  l      Lower bound
 *  \param  h      Higher bound
 *  \param  scale  Precision factor
 *
 *  \return X ~ U(l, h) with precision of scale \c scale
 */
real_t rand_real(
    real_t   l,
    real_t   h,
    unsigned scale = MATH_DEFAULT_RAND_SCALE)
{
    assert(l < h);
    assert(scale);

    const real_t real_scale(scale);

    const real_t i((real_t::impl_t)::rand() / (real_t::impl_t)INT_MAX);
    const real_t s((i * real_scale).trunc() / real_t(scale));
    const real_t x((h - l) * s + l);

    assert(l <= x && x <= h);

    return x;
}


/**
 *  \brief  Random real vector
 *
 *  \param  rank   Vector rank
 *  \param  norm   Vector norm
 *  \param  no_0s  Zero items are unacceptable
 *  \param  scale  Precision factor
 *
 *  \return Random vector with requested norm
 */
real_vector rand_real_vector(
    size_t   rank,
    real_t   norm  = 1.0,
    bool     no_0s = false,
    unsigned scale = MATH_DEFAULT_RAND_SCALE)
{
    assert(norm >= 0.0);

    real_vector x(rank);

    real_t x_norm2(0);

    for (size_t i = 0; i < rank; ++i) {
        real_t x_i;
        do x_i = rand_real(-1.0, 1.0, scale);
        while (no_0s && 0.0 == x_i);

        x_norm2 += x_i * x_i;

        x[i] = x_i;
    }

    real_t f = sqrt(norm * norm / x_norm2);
    for (size_t i = 0; i < rank; ++i) {
        x[i] *= f;
    }

    return x;
}


/**
 *  \brief  Random probability
 *
 *  Provides probability with bounded precision.
 *
 *  \param  scale  Precision factor
 */
inline real_t rand_p(unsigned scale = MATH_DEFAULT_RAND_SCALE) {
    return rand_real(0.0, 1.0, scale);
}


/**
 *  \brief  Vector of random probabilities p_i : sum p_i == 1
 *
 *  \param  rank   Vector rank
 *  \param  scale  Precision factor
 */
real_vector rand_p_vector(
    size_t rank, unsigned scale = MATH_DEFAULT_RAND_SCALE)
{
    real_vector p(rank);

    real_t h = 1.0;

    size_t i;
    for (i = 0; i < rank - 1; ++i) {
        real_t p_i = rand_real(0.0, h, scale);

        p[i] = p_i;
        h -= p_i;
    }

    p[i] = h;

    return p;
}

}  // end of namespace math

#endif  // end of #ifndef math__rng_hxx
