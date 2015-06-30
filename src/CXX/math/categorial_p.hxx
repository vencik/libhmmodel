#ifndef math__categorial_p_hxx
#define math__categorial_p_hxx

/**
 *  \brief  Random variable with categorial probability distribution
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

#include <algorithm>
#include <cassert>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

/**
 *  \brief  Selection based on probability
 *
 *  \c I must implement \c for_each generic algoritmus (i.e. taking injection)
 *  over T values providing each value and its probability.
 *
 *  \tparam T    Selected value type
 *  \tparam I    Injection type
 *  \param  v    Vector for selection (if user have its own)
 *  \param  inj  Injection (provided with \c T instance,
 *               it returns its probability)
 *
 *  \return  Selected T instance and its probability
 */
template <typename T, class I>
const T select_p(std::vector<T> & v, I inj) {
    size_t scale  = v.size();
    size_t offset = 0;

    inj.for_each([&](T t, const real_t & p) {
        size_t end = offset + (real_t(scale) * p).round();

        for (size_t i = offset; i < end && i < scale; ++i)
            v[i] = t;

        offset = end;
    });

    assert(offset == scale);

    offset = (real_t(scale - 1) * rand_p()).trunc();

    assert(offset < v.size());

    return v[offset];
}


/**
 *  \brief  Selection based on probability
 *
 *  \tparam T    Selected value type
 *  \tparam I    Injection type
 *  \param  d_v  Default value type
 *  \param  inj  Injection (provided with state reference,
 *               returns T instance and probability in std::pair)
 *
 *  \return  Selected T instance and its probability
 */
template <typename T, class I>
inline const T select_p(const T & d_v, I inj) {
    std::vector<T> v(MATH_DEFAULT_RAND_SCALE, d_v);

    return select_p(v, inj);
}


/**
 *  \brief  Selection based on probability
 *
 *  \tparam T    Selected value type
 *  \tparam I    Injection type
 *  \param  inj  Injection (provided with state reference,
 *               returns T instance and probability in std::pair)
 *
 *  \return  Selected T instance and its probability
 */
template <typename T, class I>
inline const T select_p(I inj) {
    return select_p(T(), inj);
}


/** Random variable with categorial probability distribution */
template <typename X>
class categorial_p {
    private:

    /** Probability table */
    std::map<X, real_t> m_tab;

    public:

    /** Probability getter */
    inline real_t operator () (const X & x) const {
        auto i = m_tab.find(x);

        return m_tab.end() == i ? real_t(0) : i->second;
    }

    /** Table size */
    inline size_t size() const { return m_tab.size(); }

    /** Reset all probabilities to 0.0 */
    inline void reset() { m_tab.clear(); }

    /**
     *  \brief  Loop through values
     *
     *  The injection functor takes 2 arguments:
     *  \code
     *  void operator () (const X & x, real_t p);
     *  \endcode
     *
     *  \tparam I    Injection type
     *  \param  inj  Injection
     */
    template <class I>
    inline void for_each(I inj) const {
        std::for_each(m_tab.begin(), m_tab.end(),
        [&](const std::pair<X, real_t> x) {
            inj(x.first, x.second);
        });
    }

    /** Random value getter */
    inline X rand() const {
        return select_p(X::value(0), *this);
    }

    /** Probability setter */
    inline void set(const X & x, real_t p) { m_tab[x] = p; }

};  // end of template class categorial_p

}  // end of namespace math

#endif  // end of #ifndef math__categorial_p_hxx
