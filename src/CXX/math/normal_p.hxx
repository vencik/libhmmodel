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
 *  and getting dimension using \c size() method.
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
 */
template <typename X>
class normal_p {
    private:

    X m_u;       /**< Mean                       */
    X m_sigma2;  /**< Covariance matrix diagonal */

    // Pre-computed values */
    double m_c_inv;  /**< Const. factor (inverted)          */
    X      m_A;      /**< Cov. matrix eigenvalues^1/2 for I */

    public:

    /**
     *  \brief  Constructor of standard normal distribution N(0,1)
     */
    normal_p(): m_u(0.0), m_sigma2(1.0) {
        m_A.reserve(m_u.size());
    }

    /** Constructor */
    normal_p(const X & u, const X & sigma2):
        m_u(u), m_sigma2(sigma2)
    {
        if (m_u.size() != m_sigma2.size())
            throw std::logic_error(
                "gaussian_hmm_emission_p:: invalid arguments");

        m_A.reserve(m_u.size());
    }

    /** Probability getter */
    double operator () (const X & x) const {
        assert(x.size() == m_u.size());

        // -1/2 * differenceT X Covariance matrix inversion X difference
        // Note that the matrix is assumed to be diagonal,
        // so inversion is also diagonal and composed by inversions of
        // the matrix diagonal items
        double exp_arg = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double d_i = x[i] - m_u[i];  // difference (i-th item)

            exp_arg -= d_i * d_i / m_sigma2[i];
        }
        exp_arg /= 2;

        // Density
        return ::exp(exp_arg) / m_c_inv;
    }

    /**
     *  \brief  Random value getter
     *
     *  The function uses the Central Limit Theorem to obtain vector
     *  of rand. values from unit normal distribution N(0,1) transforming
     *  vector of rand. values from unit uniform distribution U(0,1).
     *  Then, since the co-variance matrix is diagonal, it's own diagonal
     *  items act as eigenvalues for unit eigenvactors
     *  (i.e. Sigma2 i == sigma2_i i for each column i from matrix 1).
     *  Let matrix A = Sigma2^(1/2) (so that Sigma2 = A A^T).
     *  Now, if y ~ N(0,1) then linear transformation
     *  x = A y + u ~ N(u,Sigma2).
     */
    X rand() const {
        X x;

        for (size_t i = 0; i < m_u.size(); ++i) {
            // Central Limit Theorem
            // lim_{n->inf} (sum_{i=0}^n U(0, 1) - n/2) / (n/12)^1/2 ~ N(0, 1)
            // This function uses 12 random variables for the approximation
            double x_i = -6.0;  // pre-subtraction of 12/2

            // Addition of the uniform rand. vals
            for (size_t j = 0; j < 12; ++j)
                x_i += math::rand(0.0, 1.0);

            // That's it, since (12/12)^1/2 == 1

            // Transform
            x_i *= m_A[i];
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
        static const double pi_x2 = 2.0 * M_PI;  // 2 * pi

        if (u.size() != sigma2.size() || u.size() != m_u.size())
            throw std::logic_error(
                "gaussian_hmm_emission_p::set: invalid arguments");

        m_u      = u;
        m_sigma2 = sigma2;

        // (2*pi)^k * covariance matrix determinant
        // Note that the matrix is assumed to be diagonal,
        // so determinant is simply multiplication of its diagonal items
        double pi_x2_kk_x_det_sigma2 = 1.0;
        for (size_t i = 0; i < m_sigma2.size(); ++i) {
            if (m_sigma2[i] <= 0.0)
                throw std::runtime_error(
                    "gaussian_hmm_emission_p::set: invalid covariance matrix");

            pi_x2_kk_x_det_sigma2 *= pi_x2 * m_sigma2[i];

            m_A[i] = ::sqrt(m_sigma2[i]);
        }

        m_c_inv = ::sqrt(pi_x2_kk_x_det_sigma2);
    }

};  // end of template class normal_p

}  // end of namespace math

#endif  // end of #ifndef math__normal_p_hxx
