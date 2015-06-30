#ifndef math__normal_p_hxx
#define math__normal_p_hxx

/**
 *  \brief  Random variable with normal probability distribution
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

#include "math/numerics.hxx"
#include "math/rng.hxx"

#include <cassert>
#include <cmath>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

/**
 *  \brief  Random variable with normal probability distribution
 *
 *  The implementation assumes \c X to be a vector type of constant
 *  dimension, allowing for item acces via [] operator, copying
 *  and getting rank using \c rank() static and object methods.
 *  Default constructor should give zero vector.
 *
 *  Multivariate normal distribution density function is computed.
 *  For simplicity in practice, however, the covariance matrix
 *  is assumed to be diagonal.
 *  Diagonal covariance matrix means that items of the vector
 *  have 0 mutual covariance (i.e. are completely unrelated random
 *  variables).
 *  This is fine for most use-cases (and hugely simplifies the
 *  computation).
 *
 *  \tparam  X  Random variable type
 *  \tparam  F  Probability factorisation (4F sigma-derived classes)
 */
template <typename X, size_t F = 4>
class normal_p {
    public:

    /**
     *  \brief  Table of probabilities
     *
     *  Probability of P(X == x) == 0 for continuous
     *  random variable X.
     *
     *  This table, however, provides approximation of probability
     *  that X lies within an interval of size sigma/F that lies
     *  c * sigma/F far from the mean, i.e.
     *  P =
     *    P(c * sigma/F <= |X - u| < (c+1) * sigma/F) for c = 0..3F-1,
     *    P(|X - u| >= c * sigma/F) for c = 3F
     *  (where the last interval is actually prolonged to infinity).
     *
     *  3F+1 intervals are used; that is 3F sigma/F intervals
     *  and "the rest".
     *  The idea is that 3 std. deviations far from mean,
     *  the probability is low enough not to matter much any more.
     *
     *  Note that there's no need to compute the table for each
     *  N(u, sigma^2); since the factors are defined by std. deviation
     *  fraction, the probabilities will be the same for any
     *  parameters.
     *  Therefore, the table simply uses N(0, 1) (multivariate)
     *  distribution.
     */
    class table {
        private:

        std::vector<real_t> m_impl;  /**< Implementation */

        /**
         *  \brief  N(0, 1) probability density function
         *
         *  Probability density of multivariate random variable
         *  with normal probability distribution.
         *  Here, 0 and 1 mean 0-vector and unit matrix of rank D.
         *
         *  \param  x  Random vector X ~ N(0, 1)
         */
        static real_t densityN01(const X & x) {
            //static const real_t pi_x2 = 2.0 * M_PI;  // 2 * pi

            // (2*pi)^d * covariance matrix determinant
            // Note that the matrix is assumed to be diagonal,
            // so determinant is simply multiplication
            // of its diagonal items
            //
            // Furthermore, since covariance matrix is unit matrix,
            // the determinant is therefore 1
            //
            // For our purposes, however, the const. factor is not
            // imprtant, since we normalise the density to produce
            // probability, anyway.
            // Keeping the code for reference, commented-out.
            //real_t pi_x2_dd = 1.0;

            // -1/2 * deltaT X Covariance matrix inversion X delta
            // where delta = x - u.
            //
            // Note that the matrix is assumed to be diagonal,
            // so inversion is also diagonal and composed by
            // inversions of the matrix diagonal items.
            //
            // Since u = 0 and sigma = sigma^2 = 1, the computation
            // is simplified to -1/2 * x^2
            real_t exp_arg = 0.0;

            // Compute the above
            for (size_t i = 0; i < x.rank(); ++i) {
                //pi_x2_dd *= pi_x2;
                exp_arg  -= x[i] * x[i];
            }

            exp_arg /= 2;

            // Compute the density
            return exp(exp_arg) /* / sqrt(pi_x2_dd) */;
        }

        /**
         *  \brief  Table initialisation
         *
         *  Recurrent function that sets the table items to densities.
         *  Basically, it implements nested loops over dimensions.
         *
         *  \param  delta  |X - u|
         *  \param  i      Table index
         *  \param  d      Dimension index
         *
         *  \return Sum of all item values (i.e. density sum).
         */
        real_t init(const X & delta, size_t i = 0, size_t d = 0) {
            // x is ready, compute density
            if (!(d < delta.rank()))
                return m_impl[i] = densityN01(delta);

            // Set densities over dimension d
            i *= 3 * F + 1;

            X d_delta(delta);

            real_t sum_dens = 0.0;
            for (size_t c = 0; c < 3 * F + 1; ++c) {
                sum_dens += init(d_delta, i + c, d + 1);

                d_delta[d] += 1.0 / F;
            }

            return sum_dens;
        }

        public:

        /**
         *  \brief  Constructor
         */
        table() {
            // Table size: (3F + 1)^D
            size_t size = 1;

            // Compute the above
            for (size_t d = 0; d < X::rank(); ++d)
                size *= 3 * F + 1;

            // Initialise table by densities
            m_impl.assign(size, 0);

            const real_t sum_dens = init(X(0));

            // Weight densities to get probabilities
            std::for_each(m_impl.begin(), m_impl.end(),
            [&sum_dens](real_t & p) {
                p /= sum_dens;
            });
        }

        /**
         *  \brief  Probability getter
         *
         *  \param  x  X ~ N(0, 1)
         *
         *  \return Approximation of P(x -> u)
         */
        real_t operator () (const X & x) const {
            size_t i = 0;  // table index

            size_t d = 0;  // dimension
            do {
                i *= 3 * F + 1;

                // sigma/F category for dimension d
                size_t c =
                    (size_t)((x[d] /* - 0 */).abs() /* / 1 */ * real_t(F));
#if (0)
std::cerr
    << "sigma/" << F << " category in dimension " << d << ": "
    << c << std::endl;
#endif

                // Too far is far enough
                if (3 * F < c) c = 3 * F;

                i += c;

            } while (++d < x.rank());

            assert(i < m_impl.size());
//std::cerr << "Index for " << x << ": " << i << std::endl;
            return m_impl[i];
        }

    };  // end of class table

    private:

    static const table s_table;  /**< Probability table */

    X m_u;       /**< Mean                       */
    X m_sigma2;  /**< Covariance matrix diagonal */

    // Pre-computed values */
    X m_sigma;   /**< Variance */

    /**
     *  \brief  Precomputation
     *
     *  Computes covariance matrix eigenvalues squared,
     *  constant factor for probability density function and
     *  probability table.
     */
    void precompute() {
        // Variance
        // Also covariance matrix eigenvalues^1/2 for I
        for (size_t i = 0; i < m_sigma2.rank(); ++i) {
            if (m_sigma2[i] <= 0.0)
                throw std::runtime_error(
                    "gaussian_hmm_emission_p::set: invalid covariance matrix");

            m_sigma[i] = sqrt(m_sigma2[i]);
        }
    }

    public:

    /**
     *  \brief  Constructor of standard normal distribution N(0,1)
     */
    normal_p(): m_u(0.0), m_sigma2(1.0) {
        precompute();
    }

    /** Constructor */
    normal_p(const X & u, const X & sigma2):
        m_u(u), m_sigma2(sigma2)
    {
        if (m_u.rank() != m_sigma2.rank())
            throw std::logic_error(
                "gaussian_hmm_emission_p: invalid arguments");

        precompute();
    }

    /** Probability getter */
    real_t operator () (const X & x) const {
        // Transform X ~ N(u, sigma^2) to Y ~ N(0, 1)
        // so that y[d] = (x[d] - u[d]) / sigma[d]
        // is random variable that evaluates to distance of x from u
        // in sigma units.
        //
        // Observe that mean of Y is 0 (i.e. zero distance) and
        // variance of Y is 1 (unit).
        X y(x);
        for (size_t d = 0; d < x.rank(); ++d) {
            y[d] -= m_u[d];
            y[d] /= m_sigma[d];
        }
#if (0)
std::cerr
    << "x == " << x << ", y == " << y << ", sigma == " << m_sigma
    << std::endl;
#endif

        // Distribution function of Y (in any dimension):
        // phi(y) = (1/sqrt(2pi)) * exp(-1/2 * (x - u)/sigma)
        // differs from phi(x) only by the const. factor,
        // which is unimportant for the computation anyway, since
        // it's overriden by the normalisation (i.e.
        // p_i = (a * pi_i) / sum_j (a * pi_j)
        // p_i = (a * pi_i) / a * sum_j pi_j
        // p_i = pi_i / sum_j pi_j
        // where pi_i = phi(x) for x being a representant of each
        // discretisation class).
        //
        // Thus, fetching the discretised probability of Y
        // gives the result.
        real_t p = s_table(y);

#if (1)
std::cerr
    << "P(X ~ N(" << m_u << ", " << m_sigma2 << ") =~ "
    << x << ") == " << p << std::endl;
#endif
        return p;
    }

    /**
     *  \brief  Random value getter
     *
     *  The function uses the Central Limit Theorem to obtain vector
     *  of rand. values from unit normal distribution N(0,1) transforming
     *  vector of rand. values from unit uniform distribution U(0,1).
     *  Then, since the co-variance matrix is diagonal, it's own diagonal
     *  items act as eigenvalues for unit eigenvectors
     *  (i.e. Sigma2 i == sigma2_i i for each column i from matrix 1).
     *  Let matrix A = Sigma2^(1/2) (so that Sigma2 = A A^T).
     *  Now, if y ~ N(0,1) then linear transformation
     *  x = A y + u ~ N(u,Sigma2).
     */
    X rand() const {
        X x;

        for (size_t i = 0; i < m_u.rank(); ++i) {
            // Central Limit Theorem
            // lim_{n->inf} (sum_{i=0}^n U(0, 1) - n/2) / (n/12)^1/2 ~ N(0, 1)
            // This function uses 12 random variables for the approximation
            real_t x_i = -6.0;  // pre-subtraction of 12/2

            // Addition of the uniform rand. vals
            for (size_t j = 0; j < 12; ++j)
                x_i += math::rand_real(0.0, 1.0);

            // That's it, since (12/12)^1/2 == 1

            // Transform
            x_i *= m_sigma[i];
            x_i += m_u[i];

            x[i] = x_i;
        }

        return x;
    }

    /** Average getter */
    inline const X & u() const { return m_u; }

    /** Covariance matrix diagonal getter */
    inline const X & sigma2() const { return m_sigma2; }

    /**
     *  \brief  Distribution function parameters setter
     *
     *  Throws exception if any covariance matrix diagonal item is <= 0.
     *
     *  \param  u       Mean
     *  \param  sigma2  Covariance matrix diagonal (variance in 1D case)
     */
    void set(const X & u, const X & sigma2) throw(std::runtime_error) {
        if (u.rank() != sigma2.rank() || u.rank() != m_u.rank())
            throw std::logic_error(
                "gaussian_hmm_emission_p::set: invalid arguments");

        m_u      = u;
        m_sigma2 = sigma2;

        precompute();
    }

};  // end of template class normal_p

// Static members initialisation
template <typename X, size_t F>
const typename normal_p<X, F>::table normal_p<X, F>::s_table;

}  // end of namespace math

#endif  // end of #ifndef math__normal_p_hxx
