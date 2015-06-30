#ifndef math__numerics_hxx
#define math__numerics_hxx

/**
 *  Base numerics
 *
 *  \date    2015/06/27
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


#include <stdexcept>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>


namespace math {

/**
 *  \brief  Base numeric type for R
 */
template <typename R>
class real {
    public:

    typedef R impl_t;  /**< Impementation type */

    protected:

    impl_t m_impl;  /**< Implementation */

    public:

    /** Default constructor (0) */
    real(): m_impl(0.0) {}

    /**
     *  \brief  Constructor
     *
     *  \param  init  Initialiser
     */
    real(const impl_t & init): m_impl(init) {}

    /** Conversion to impl. type */
    const impl_t & get_impl() const { return m_impl; }
    //operator impl_t () const { return m_impl; }

    // Algebraic operations
    /** \cond */
    real operator + () const { return  m_impl; }
    real operator - () const { return -m_impl; }

    real & operator += (const real & rarg) {
        m_impl += rarg.m_impl;
        return *this;
    }

    real & operator -= (const real & rarg) {
        m_impl -= rarg.m_impl;
        return *this;
    }

    real & operator *= (const real & rarg) {
        m_impl *= rarg.m_impl;
        return *this;
    }

    real & operator /= (const real & rarg) {
        m_impl /= rarg.m_impl;
        return *this;
    }

    real operator + (const real & rarg) const {
        real sum(m_impl);
        return sum += rarg;
    }

    real operator - (const real & rarg) const {
        real sub(m_impl);
        return sub -= rarg;
    }

    real operator * (const real & rarg) const {
        real pro(m_impl);
        return pro *= rarg;
    }

    real operator / (const real & rarg) const {
        real div(m_impl);
        return div /= rarg;
    }

    real sqrt() const { return ::sqrt(m_impl); }

    real exp() const { return ::exp(m_impl); }
    /** \endcond */

    // Comparisons
    /** \cond */
    bool operator == (const impl_t & rarg) const { return m_impl == rarg; }
    bool operator != (const impl_t & rarg) const { return m_impl != rarg; }

    bool operator == (const real & rarg) const { return *this == rarg.m_impl; }
    bool operator != (const real & rarg) const { return *this != rarg.m_impl; }

    bool operator <= (const impl_t & rarg) const { return m_impl <= rarg; }
    bool operator >= (const impl_t & rarg) const { return m_impl >= rarg; }

    bool operator <= (const real & rarg) const { return *this <= rarg.m_impl; }
    bool operator >= (const real & rarg) const { return *this >= rarg.m_impl; }

    bool operator < (const impl_t & rarg) const { return m_impl < rarg; }
    bool operator > (const impl_t & rarg) const { return m_impl > rarg; }

    bool operator < (const real & rarg) const { return *this < rarg.m_impl; }
    bool operator > (const real & rarg) const { return *this > rarg.m_impl; }
    /** \endcond */

    /** Base type assignment */
    real & operator = (const impl_t & rarg) {
        m_impl = rarg;
        return *this;
    }

    /** Absolute value */
    real abs() const {
        if (m_impl < 0) return -m_impl;

        return m_impl;
    }

    /** Truncate (chop non-integer part off) */
    real trunc() const { return (impl_t)(long long)m_impl; }

    /** Round (to closest integer) */
    real round() const {
        impl_t s = m_impl < 0 ? -1 : 1;
        impl_t i(s * m_impl);

        if ((long long)i < (long long)(i + 0.5))
            i += 1;

        return s * i;
    }

    /** Truncate and convert to integer */
    operator long long () const { return (long long)m_impl; }

};  // end of template class real

// Comparisons
/** \cond */
template <typename R>
bool operator == (const typename real<R>::impl_t & larg, const real<R> & rarg) {
    return rarg == larg;  // == is comutative
}

template <typename R>
bool operator != (const typename real<R>::impl_t & larg, const real<R> & rarg) {
    return rarg != larg;  // != is comutative
}

template <typename R>
bool operator <= (const typename real<R>::impl_t & larg, const real<R> & rarg) {
    return rarg >= larg;
}

template <typename R>
bool operator >= (const typename real<R>::impl_t & larg, const real<R> & rarg) {
    return rarg <= larg;
}

template <typename R>
bool operator < (const typename real<R>::impl_t & larg, const real<R> & rarg) {
    return rarg > larg;
}

template <typename R>
bool operator > (const typename real<R>::impl_t & larg, const real<R> & rarg) {
    return rarg < larg;
}
/** \endcond */


/** Square root */
template <typename R>
real<R> sqrt(const real<R> & arg) { return arg.sqrt(); }

/** e^x */
template <typename R>
real<R> exp(const real<R> & arg) { return arg.exp(); }


/** Real number serialisation */
template <typename R>
std::ostream & operator << (std::ostream & out, const real<R> & rnum) {
    out << rnum.get_impl();
    return out;
}

/** Real number deserialisation */
template <typename R>
std::istream & operator >> (std::istream & in, real<R> & rnum) {
    typename real<R>::impl_t impl;
    in >> impl;
    rnum = impl;
    return in;
}


/**
 *  \brief  Vector
 *
 *  \tparam M Base numeric type
 */
template <typename M>
class vector {
    public:

    typedef M base_t;  /**< Base numeric type */

    private:

    std::vector<base_t> m_impl;  /**< Implementation */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  rank  Rank
     */
    vector(size_t rank): m_impl(rank, base_t()) {}

    /**
     *  \brief  Constructor
     *
     *  \param  rank  Rank
     *  \param  init  Initialiser
     */
    vector(size_t rank, const base_t & init):
        m_impl(rank, init)
    {}

    /**
     *  \brief  Constructor
     *
     *  \param  init_list  Initialiser list
     */
    vector(const std::initializer_list<base_t> & init_list):
        m_impl(init_list)
    {}

    /** Vector rank */
    size_t rank() const { return m_impl.size(); }

    /** Access operator */
    base_t & operator [] (size_t i) { return m_impl[i]; }

    /** Access operator (const) */
    const base_t & operator [] (size_t i) const { return m_impl[i]; }

    /**
     *  \brief  Multiplication by scalar (in place)
     *
     *  \param  coef  Scalar coefficient
     */
    vector & operator *= (const base_t & coef) {
        std::for_each(m_impl.begin(), m_impl.end(),
        [coef](base_t & item) {
            item *= coef;
        });

        return *this;
    }

    /**
     *  \brief  Multiplication by scalar
     *
     *  \param  coef  Scalar coefficient
     */
    vector operator * (const base_t & coef) const {
        vector result(*this);
        result *= coef;
        return result;
    }

    /**
     *  \brief  Division by scalar (in place)
     *
     *  \param  denom  Scalar denominator
     */
    vector & operator /= (const base_t & denom) { return *this *= 1/denom; }

    /**
     *  \brief  Division by scalar
     *
     *  \param  denom  Scalar denominator
     */
    vector operator / (const base_t & denom) const { return *this * 1/denom; }

    /**
     *  \brief  Scalar multiplication
     *
     *  \param  rarg  Right-hand argument
     */
    base_t operator * (const vector & rarg) const {
        if (m_impl.size() != rarg.m_impl.size())
            throw std::logic_error(
                "math::vector: incompatible scalar mul. args");

        base_t sum = 0;
        for (size_t i = 0; i < m_impl.size(); ++i)
            sum += m_impl[i] * rarg.m_impl[i];

        return sum;
    }

    private:

    /**
     *  \brief  Per-item operation (in place)
     *
     *  \param  rarg  Right-hand argument
     *  \param  fn    Item computation
     */
    template <class Fn>
    vector & per_item_in_place(const vector & rarg, Fn fn) {
        if (m_impl.size() != rarg.m_impl.size())
            throw std::logic_error(
                "math::vector: incompatible args");

        for (size_t i = 0; i < m_impl.size(); ++i)
            m_impl[i] = fn(m_impl[i], rarg[i]);

        return *this;
    }

    public:

    /** Vector addition (in place) */
    vector & operator += (const vector & rarg) {
        return per_item_in_place(rarg,
        [](const base_t & la, const base_t & ra) {
            return la + ra;
        });
    }

    /** Vector addition */
    vector operator + (const vector & rarg) const {
        vector result(*this);
        result += rarg;
        return result;
    }

    /** Vector subtraction (in place) */
    vector & operator -= (const vector & rarg) {
        return per_item_in_place(rarg,
        [](const base_t & la, const base_t & ra) {
            return la - ra;
        });
    }

    /** Vector subtraction */
    vector operator - (const vector & rarg) const {
        vector result(*this);
        result -= rarg;
        return result;
    }

    /** Vectors equal */
    bool operator == (const vector & rarg) const {
        if (m_impl.size() != rarg.m_impl.size())
            throw std::logic_error(
                "math::vector: comparing incompatible args");

        for (size_t i = 0; i < m_impl.size(); ++i)
            if (m_impl[i] != rarg.m_impl[i]) return false;

        return true;
    }

    /** Vectors not equal */
    bool operator != (const vector & rarg) const {
        return !(*this == rarg);
    }

};  // end of template class vector


/** Product of scalar and vector */
template <typename M>
vector<M> operator * (const M & larg, const vector<M> & rarg) {
    return rarg * larg;  // c * v is comutative
}


/** Vector serialisation */
template <typename M>
std::ostream & operator << (std::ostream & out, const vector<M> & vec) {
    out << '[';

    for (size_t i = 0; i < vec.rank() - 1; ++i)
        out << vec[i] << ' ';

    out << vec[vec.rank() - 1] << ']';

    return out;
}

/** Vector deserialisation */
template <typename M>
std::istream & operator >> (std::istream & in, vector<M> & vec) {
    char c;

    in >> c;
    if ('[' != c)
        throw std::runtime_error(
            "math::vector: parse error: '[' expected");

    for (size_t i = 0; i < vec.rank(); ++i)
        in >> vec[i];

    in >> c;
    if (']' != c)
        throw std::runtime_error(
            "math::vector: parse error: ']' expected");

    return in;
}


/**
 *  \brief  Real numbers
 *
 *  Double precision should do for now.
 */
typedef real<double> real_t;


/** Real vectors */
typedef vector<real_t> real_vector;

}  // end of namespace math

#endif  // end of #ifndef math__numerics_hxx
