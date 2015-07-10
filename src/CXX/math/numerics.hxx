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
#include <cassert>


namespace math {

/**
 *  \brief  significand * 2^exponent
 *
 *  Representation of real numbers in form of
 *      significand * 2^exponent
 *
 *  where 0.5 <= significand < 1
 *
 *  \tparam  Significand  Significand type (floating point type)
 *  \tparam  Exponent     Exponent type (singed integer type)
 *  \tparam  Packed       Packed type (floating point type)
 */
template <
    typename Significand,
    typename Exponent,
    typename Packed = long double>
class sign2exp {
    public:

    typedef Significand sign_t;  /**< Significand type (floating point) */
    typedef Exponent    exp_t;   /**< Exponent type (integer)           */
    typedef Packed      pack_t;  /**< Packed type (floating point)      */

    protected:

    sign_t m_sign;  /**< Significand */
    exp_t  m_exp;   /**< Exponent    */

    /** Transform from packed to sign*2^exp */
    void unpack(pack_t pack) { m_sign = ::frexp(pack, &m_exp); }

    /** Transform from sign*2^exp to packed */
    pack_t pack() const { return ::ldexp(m_sign, m_exp); }

    /** Normalise 0 */
    void normalise_0() { if (0.0 == m_sign) m_exp = 0; }

    /** Normalise low significand */
    void normalise_low() {
        while (0.0 != m_sign && ::fabs(m_sign) < 0.5) {
            m_sign *= 2.0;
            m_exp  -= 1;
        }
    }

    /** Normalise high significand */
    void normalise_high() {
        while (1.0 <= ::fabs(m_sign)) {
            m_sign /= 2.0;
            m_exp  += 1;
        }
    }

    /**
     *  \brief  Normalise
     *
     *  Normalise significand so that it's in [0.5, 1)
     *  (except for 0.0).
     */
    void normalise() {
        normalise_0();
        normalise_low();
        normalise_high();

        assert(
            (0.0 == m_sign && 0 == m_exp) ||
            (0.5 <= ::fabs(m_sign) && ::fabs(m_sign) < 1.0));
    }

    /**
     *  \brief  Denormalise
     *
     *  NOTE THAT denormalisation may end up in loosing significand bits.
     *
     *  \param  shift  Exponent shift
     */
    void denormalise(exp_t shift) {
        m_exp += shift;

        if (0 > shift)
            while (shift++) m_sign *= 2.0;
        else
            while (shift--) m_sign /= 2.0;
    }

    /**
     *  \brief  Constructor (by members, internal)
     *
     *  NOTE: The constructor DOESN'T NORMALISE the arguments!
     *
     *  \param  sign  Significand
     *  \param  exp   Exponent
     */
    sign2exp(const sign_t & sign, const exp_t & exp):
        m_sign(sign),
        m_exp(exp)
    {}

    public:

    /** Default constructor (0) */
    sign2exp(): m_sign(0.0), m_exp(0) {}

    /**
     *  \brief  Constructor
     *
     *  \param  init  Initialiser
     */
    sign2exp(const pack_t & init) { unpack(init); }

    /** Significand getter */
    const sign_t & significand() const { return m_sign; }

    /** Exponent getter */
    const exp_t & exponent() const { return m_exp; }

    /** Assignment of packed value */
    sign2exp & operator = (const pack_t & rarg) {
        unpack(rarg);
        return *this;
    }

    /** Unary + */
    sign2exp operator + () const { return sign2exp(*this); }

    /** Unary - */
    sign2exp operator - () const {
        sign2exp min_this(*this);

        // Avoid -0.0
        if (0.0 != min_this.m_sign)
            min_this.m_sign = -min_this.m_sign;

        return min_this;
    }

    /**
     *  \brief  Multiplication (in place)
     *
     *  Multiplication is numerically safe except for overflow.
     *
     *  x * y = sign_x * 2^exp_x * sign_y * 2^exp_y
     *        = sign_x * sign_y * 2^(exp_x + exp_y)
     *
     *  0.25 <= |sign_x * sign_y| < 1
     *
     *  \param  rarg  Right argument
     */
    sign2exp & operator *= (const sign2exp & rarg) {
        if (0.0 == m_sign) return *this;  // 0 * x == 0

        m_sign *= rarg.m_sign;
        m_exp  += rarg.m_exp;

        normalise_0();
        normalise_low();

        return *this;
    }

    /** Multiplication */
    sign2exp operator * (const sign2exp & rarg) const {
        return sign2exp(*this) *= rarg;
    }

    /**
     *  \brief  Division (in place)
     *
     *  Division is numerically safe except for underflow.
     *
     *  x / y = sign_x * 2^exp_x / (sign_y * 2^exp_y)
     *        = sign_x / sign_y * 2^(exp_x - exp_y)
     *
     *  0.5 <= |sign_x / sign_y| < 2
     *
     *  \param  rarg  Right argument
     */
    sign2exp & operator /= (const sign2exp & rarg) {
        if (0.0 == m_sign) return *this;  // 0 / x == 0

        // Division by 0
        if (0.0 == rarg.m_sign)
            throw std::runtime_error(
                "math::sign2exp: division by 0");

        m_sign /= rarg.m_sign;
        m_exp  -= rarg.m_exp;

        normalise_0();  // 0 means underflow!
        normalise_high();

        return *this;
    }

    /** Division */
    sign2exp operator / (const sign2exp & rarg) const {
        return sign2exp(*this) /= rarg;
    }

    /**
     *  \brief  Inversion
     *
     *  x^-1 = (sign_x * 2^exp_x)^-1 = sin_x^-1 * (2^exp_x)^-1
     *       = sign_x^-1 * 2^-exp_x
     *
     *  0.5 <= |sign_x| < 1 => 1 < |sign_x^-1| <= 2
     *  This means that normalisation by factor 1/2 is mandatory
     *  and if 1 = sign_x^-1 then another 1/2 normalisation is required.
     *
     *  \param  rarg  Right argument
     */
    sign2exp inv() const {
        // Division by 0
        if (0.0 == m_sign)
            throw std::runtime_error(
                "math::sign2exp: inversion of 0");

        sign2exp i((1.0 / m_sign) / 2.0, -m_exp + 1);

        if (1.0 == i.m_sign) {
            i.m_sign /= 2.0;
            i.m_exp  += 1;
        }

        // Underflow
        else if (0.0 == i.m_sign)
            i.normalise_0();

        return i;
    }

    private:

    /**
     *  \brief  Addition implementation (in place)
     *
     *  It is assumed that \c rarg exponent is greater or equal to \c this.
     *  Then
     *  x + y = sign_x * 2^exp_x + sign_y * 2^exp_y
     *        = 2^exp_x * (sign_x + sign_y * 2^(exp_y - exp_x))
     *
     *  -2 < sign_x + sign_y < 2
     *
     *  \param  rarg  Right argument
     */
    sign2exp & add_impl(const sign2exp & rarg) {
        exp_t exp_diff = rarg.m_exp - m_exp;

        assert(0 <= exp_diff);

        sign_t rarg_sign_2exp_diff = rarg.m_sign;
        while (exp_diff--)
            rarg_sign_2exp_diff *= 2.0;

        m_sign += rarg_sign_2exp_diff;

        normalise();

        return *this;
    }

    public:

    /**
     *  \brief  Addition (in place)
     *
     *  NOTE THAT addition may loose significand bits of the operand
     *  with smaller exponent if exponents difference is too high.
     *  You should never sum up too different numbers (unless compensating
     *  for the lost bits).
     *
     *  \param  rarg  Right argument
     */
    sign2exp & operator += (const sign2exp & rarg) {
        return m_exp <= rarg.m_exp
            ? add_impl(rarg)
            : *this = sign2exp(rarg).add_impl(*this);
    }

    /**
     *  \brief  Addition
     *
     *  \see operator += for info on numerical stability.
     *
     *  \param  rarg  Right argument
     */
    sign2exp operator + (const sign2exp & rarg) const {
        return m_exp <= rarg.m_exp
            ? sign2exp(*this).add_impl(rarg)
            : sign2exp(rarg).add_impl(*this);
    }

    /**
     *  \brief  Subtraction (in place)
     *
     *  \see operator += for info on numerical stability.
     *
     *  \param  rarg  Right argument
     */
    sign2exp operator -= (const sign2exp & rarg) {
        return *this += -rarg;
    }

    /**
     *  \brief  Subtraction
     *
     *  \see operator += for info on numerical stability.
     *
     *  \param  rarg  Right argument
     */
    sign2exp operator - (const sign2exp & rarg) const {
        return *this + -rarg;
    }

    private:

    /**
     *  \brief  Comparison of absolute values
     *
     *  |x| < |y| <=> sign_x * 2^exp_x < sign_y * 2^exp_y
     *
     *  The greater 2^exp, the greater |result|.
     *  The greater exp, the greater 2^exp since
     *  exponential function is monotonous and positive.
     *  The greater significand, the greater |result|.
     *
     *  \param  rarg  Right argument
     */
    bool less_abs(const sign2exp & rarg) const {
        if (m_exp != rarg.m_exp) return m_exp < rarg.m_exp;

        // Equal exponents
        return ::fabs(m_sign) < ::fabs(rarg.m_sign);
    }

    public:

    /**
     *  \brief  Comparison
     *
     *  Negative is < positive or 0.
     *  +|x| < +|y| <=> |x| < |y|
     *  -|x| < -|y| <=> |y| < |x|
     *
     *  \param  rarg  Right argument
     */
    bool operator < (const sign2exp & rarg) const {
        if (m_sign < 0.0) {
            if (rarg.m_sign >= 0.0) return true;  // - < 0+

            return rarg.less_abs(*this);  // both -
        }

        if (rarg.m_sign < 0.0) return false;  // !(0+ < -)

        return less_abs(rarg);  // both 0+
    }

    /** Comparison */
    bool operator == (const sign2exp & rarg) const {
        return m_exp == rarg.m_exp && m_sign == rarg.m_sign;
    }

    /** Comparison */
    bool operator <= (const sign2exp & rarg) const {
        return *this < rarg || *this == rarg;
    }

    /** Comparison */
    bool operator > (const sign2exp & rarg) const { return rarg <= *this; }

    /** Comparison */
    bool operator >= (const sign2exp & rarg) const { return rarg < *this; }

    /** Comparison */
    bool operator != (const sign2exp & rarg) const {
        return !(*this == rarg);
    }

    /** Absolute value */
    sign2exp abs() const { return sign2exp(::fabs(m_sign), m_exp); }

    /**
     *  \brief  Truncate
     *
     *  NOTE: For very high exponents, this may produce completely
     *  wrong result due overflow!
     */
    explicit operator long long () const { return (long long)pack(); }

    /**
     *  \brief  Square root
     *
     *  x^(1/2) = (sin_x * 2^exp_x)^(1/2) = sign_x^(1/2) * (2^exp_x)^(1/2)
     *      = sign_x^(1/2) * 2^(exp_x/2)                  for exp_x = 2n
     *      = sign_x^(1/2) * 2^(1/2) * 2^((exp_x - 1)/2)  for exp_x = 2n + 1
     *
     *  Since 0.5 <= sign_x < 1, then 0.5 <= sign_x^(1/2) < 1
     *  However, 1 <= (sign_x * 2)^(1/2) < 2^(1/2), so the second case
     *  ALWAYS needs normalising by factor of 1/2.
     */
    sign2exp sqrt() const {
        static const sign_t sqrt2_2 = M_SQRT2 / 2.0;

        if (0.0 > m_sign)
            throw std::runtime_error(
                "math::sign2exp: square root of a negative value");

        sign_t sign = ::sqrt(m_sign);
        exp_t  exp  = m_exp / 2;

        // exp = 2n + 1
        if (m_exp % 2) {
            sign *= sqrt2_2;
            exp  -= 1;
        }

        return sign2exp(sign, exp);
    }

    /**
     *  \brief  e^x
     */
    sign2exp exp() const { return exp(pack()); }  // can I do better?

    /** Serialisation */
    friend std::ostream & operator << (
        std::ostream &   out,
        const sign2exp & x)
    {
        out << x.pack();
        return out;
    }

    /** Deserialisation */
    friend std::istream & operator >> (
        std::istream & in,
        sign2exp &     x)
    {
        pack_t x_pack;
        if ((in >> x_pack).fail())
            throw std::runtime_error(
                "math::sign2exp: parse error");

        x = x_pack;
        return in;
    }

    /**
     *  \brief  Sum over a collection
     *
     *  The sum algorithm attempts to improve numeric stability by
     *  1/ sorting the summands by exponent
     *  2/ using running addition error compensation
     *
     *  Point 1/ aims at adding numbers of similar magnitude, decreasing
     *  the operation error.
     *
     *  Point 2/ attempts to eliminate even the above error
     *  (it may be obtained separately after the summation).
     *  Kahan summation algorithm is used
     *  (\see https://en.wikipedia.org/wiki/Kahan_summation_algorithm).
     *
     *  \param  summands  Container of summands
     *  \param  error     Running addition error
     *
     *  \return Sum
     */
    template <class Container>
    static sign2exp sum(const Container & summands, sign2exp & error) {
        typedef std::vector<const sign2exp *> args_t;

        // Sort summands
        args_t args;
        args.reserve(summands.size());

        std::for_each(summands.begin(), summands.end(),
        [&args](const sign2exp & arg) {
            args.push_back(&arg);
        });

        std::sort(args.begin(), args.end(),
        [](const sign2exp * larg, const sign2exp * rarg) -> bool {
            return larg->m_exp < rarg->m_exp;
        });

        // Summation with running error
        sign2exp sum;

        error = 0.0;
        std::for_each(args.begin(), args.end(),
        [&sum, &error](const sign2exp * arg) {
#if (0)
            // Kahan summation
            //
            // Commented out; it may provide good results for random
            // order of summands magnitude, but keeping the running
            // error separately seems to give better results (see below).
            auto arg_min_err = *arg - error;
            auto sum_min_err = sum + arg_min_err;

            error = (sum_min_err - sum) - arg_min_err;

            sum = sum_min_err;
#endif
            // Running error collection
            //
            // Don't attempt to add the error back; since the numbers'
            // magnitudes rise, it's pointless...
            // Better return it, separately.
            auto sum_new = sum + *arg;
            error += *arg - (sum_new - sum);
            sum = sum_new;

#if (0)
        std::cerr
            << "arg: " << *arg
            << ",\tsum: " << sum
            << ",\terror: " << error
            << std::endl;
#endif
        });

        return sum;
    }

    /**
     *  \brief  Sum over a collection
     *
     *  \see sum above.
     *
     *  \param  summands  Container of summands
     *
     *  \return Sum
     */
    template <class Container>
    static sign2exp sum(const Container & summands) {
        sign2exp err;
        sign2exp sum_min_err = sum<Container>(summands, err);

        return sum_min_err + err;
    }

};  // end of template class sign2exp

// Operations with packed value
/** \cond */
template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator += (sign2exp<S, E, P> & larg, const P & rarg) {
    return larg += sign2exp<S, E, P>(rarg);
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator -= (sign2exp<S, E, P> & larg, const P & rarg) {
    return larg -= sign2exp<S, E, P>(rarg);
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator *= (sign2exp<S, E, P> & larg, const P & rarg) {
    return larg *= sign2exp<S, E, P>(rarg);
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator /= (sign2exp<S, E, P> & larg, const P & rarg) {
    return larg /= sign2exp<S, E, P>(rarg);
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator + (const sign2exp<S, E, P> & larg, const P & rarg)
{
    return larg + sign2exp<S, E, P>(rarg);
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator + (const P & larg, const sign2exp<S, E, P> & rarg)
{
    return rarg + larg;
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator - (const sign2exp<S, E, P> & larg, const P & rarg)
{
    return larg + sign2exp<S, E, P>(-rarg);
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator - (const P & larg, const sign2exp<S, E, P> & rarg)
{
    return -rarg + larg;
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator * (const sign2exp<S, E, P> & larg, const P & rarg)
{
    return larg * sign2exp<S, E, P>(rarg);
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator * (const P & larg, const sign2exp<S, E, P> & rarg)
{
    return rarg * larg;
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator / (const sign2exp<S, E, P> & larg, const P & rarg)
{
    return larg * (1.0 / rarg);
}

template <typename S, typename E, typename P>
sign2exp<S, E, P> & operator / (const P & larg, const sign2exp<S, E, P> & rarg)
{
    return rarg.inv() * larg;
}

/** \endcond */

}  // end of namespace math


/** Absolute value */
template <typename S, typename E, typename P>
math::sign2exp<S, E, P> abs(const math::sign2exp<S, E, P> & arg) {
    return arg.abs();
}

/** Square root */
template <typename S, typename E, typename P>
math::sign2exp<S, E, P> sqrt(const math::sign2exp<S, E, P> & arg) {
    return arg.sqrt();
}

/** e^x */
template <typename S, typename E, typename P>
math::sign2exp<S, E, P> exp(const math::sign2exp<S, E, P> & arg) {
    return arg.exp();
}

#endif  // end of #ifndef math__numerics_hxx
