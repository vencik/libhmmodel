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
double rand(
    double   l,
    double   h,
    unsigned scale = MATH_DEFAULT_RAND_SCALE)
{
    assert(l < h);
    assert(scale);

    double i = (double)::rand() / (double)INT_MAX;
    double s = (unsigned)(i * scale) / (double)scale;
    double x = (h - l) * s + l;

    assert(l <= x && x <= h);

    return x;
}


/**
 *  \brief  Random vector
 *
 *  \param  n      Vector size
 *  \param  norm   Vector norm
 *  \param  scale  Precision factor
 *
 *  \return Random vector with requested norm
 */
std::vector<double> rand(
    size_t   n,
    double   norm  = 1.0,
    unsigned scale = MATH_DEFAULT_RAND_SCALE)
{
    assert(n > 0);
    assert(norm >= 0);

    std::vector<double> x;
    x.reserve(n);

    double x_norm2 = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double x_i = rand(-1.0, 1.0, scale);

        x_norm2 += x_i * x_i;

        x.push_back(x_i);
    }

    double f2 = norm * norm / x_norm2;

    for (auto x_i = x.begin(); x_i != x.end(); ++x_i)
        *x_i *= f2;

    return x;
}


/**
 *  \brief  Random probability
 *
 *  Provides probability with bounded precision.
 *
 *  \param  scale  Precision factor
 */
inline double rand_p(unsigned scale = MATH_DEFAULT_RAND_SCALE) {
    return rand(0.0, 1.0, scale);
}


/**
 *  \brief  \c n random probabilities p_i : sum p_i == 1
 *
 *  \param  n      Number of probabilities generated
 *  \param  scale  Precision factor
 */
std::vector<double> rand_p_vec(
    size_t n, unsigned scale = MATH_DEFAULT_RAND_SCALE)
{
    assert(n > 0);

    std::vector<double> p;
    p.reserve(n);

    double h = 1.0;

    for (size_t i = 0; i < n - 1; ++i) {
        double p_i = rand(0.0, h, scale);

        p.push_back(p_i);
        h -= p_i;
    }

    p.push_back(h);

    return p;
}

}  // end of namespace math

#endif  // end of #ifndef math__rng_hxx
