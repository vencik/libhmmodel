#ifndef math__hmm_hxx
#define math__hmm_hxx

/**
 *  \brief  Hidden Markov Model
 *
 *  See http://en.wikipedia.org/wiki/Hidden_Markov_model
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

#include "math/hmm_base.hxx"
#include "math/hmm_discrete.hxx"
#include "math/hmm_gaussian.hxx"
#include "math/hmm_simulation.hxx"
#include "math/forward.hxx"
#include "math/viterbi.hxx"
#include "math/baum-welch.hxx"
#include "math/numerics.hxx"

#include <iostream>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

/**
 *  \brief  Categorial emission type wrapper
 *
 *  Useful to create the actual \c Y template parameter for HMM.
 *  Certain actions (like e.g. the Baum-Welch algorithm computation)
 *  require that the emission type \c Y provides the following functions:
 *
 *  \code
 *  static size_t cardinality();
 *  static Y value(size_t k);
 *  \endcode
 *
 *  In such cases, a wrapper around the actual emission type might
 *  be required (if the type isn't a class one).
 *  This wrapper pre-defines the necessary operators.
 *
 *  \tparam  Y  The actual emission type
 */
template <typename Y>
class hmm_emission_t {
    private:

    Y m_value;  /**< Implementation */

    protected:

    public:

    /** Constructor */
    hmm_emission_t(const Y & value): m_value(value) {}

    /** Wrapped value access (read-only) */
    operator const Y & () const { return m_value; }

    /** Assignment */
    const hmm_emission_t & operator = (const hmm_emission_t & rval) {
        m_value = rval.m_value;

        return *this;
    }

    /** Comparison */
    bool operator == (const hmm_emission_t & rval) const {
        return m_value == rval.m_value;
    }

    /** Comparison */
    bool operator < (const hmm_emission_t & rval) const {
        return m_value < rval.m_value;
    }

    /** Serialisation */
    std::ostream & serialise(std::ostream & out) const {
        return out << m_value;
    }

    /** Deserialisation */
    std::istream & deserialise(std::istream & in) {
        return in >> m_value;
    }

};  // end of template class hmm_emission_t

/** Serialisation of HMM emission wrapper */
template <typename Y>
inline std::ostream & operator << (
    std::ostream & out, const hmm_emission_t<Y> & e)
{
    return e.serialise(out);
}

/** Deserialisation of HMM emission wrapper */
template <typename Y>
inline std::istream & operator >> (
    std::istream & in, hmm_emission_t<Y> & e)
{
    return e.deserialise(in);
}


/** Hidden Markov Model with categorial emissions */
template <typename X, typename Y>
class hmm: public impl::hmm<X, Y, impl::hmm_emit_categorial<Y> > {
    public:

    /** Emission probability */
    typedef typename impl::hmm_emit_categorial<Y> emission_p_t;

    /** Model state */
    typedef typename impl::hmm<X, Y, emission_p_t>::state_t state_t;

    /** Simulation */
    typedef typename impl::hmm_simulation<X, Y, emission_p_t> simulation;

    /** Forward algorithm */
    typedef typename impl::forward<X, Y, emission_p_t> forward;

    /** Viterbi algorithm */
    typedef typename impl::viterbi<X, Y, emission_p_t> viterbi;

    /** Baum-Welch algorithm */
    typedef impl::baum_welch<X, Y, impl::hmm_emit_categorial<Y>,
                             impl::baum_welch_emit_categorial<X, Y> >
        baum_welch;

    /**
     *  \brief  Set emission probability
     *
     *  \param  state  Model state
     *  \param  y      Observation value
     *  \param  p      Emission probability
     */
    inline void emission(state_t & state, const Y & y, const real_t & p) {
        state.value.emit_p.set(y, p);
    }

    /**
     *  \brief  Deserialise model from input stream
     *
     *  Throws an exception on invalid model string.
     *
     *  \param  in  Inpiut stream
     */
    hmm(std::istream & in) {
        if (!this->deserialise(in))
            throw std::runtime_error(
                "math::hmm: deserialisation failed");
    }

    protected:

    /** Default constructor */
    hmm():
        impl::hmm<X, Y, impl::hmm_emit_categorial<Y> >()
    {}

};  // end of template class hmm


/** Hidden Markov Model with Gaussian observations */
template <typename X, typename Y>
class ghmm: public impl::hmm<X, Y, impl::hmm_emit_gaussian<Y> > {
    public:

    /** Emission probability */
    typedef typename impl::hmm_emit_gaussian<Y> emission_p_t;

    /** Model state */
    typedef typename impl::hmm<X, Y, emission_p_t>::state_t state_t;

    /** Simulation */
    typedef typename impl::hmm_simulation<X, Y, emission_p_t> simulation;

    /** Forward algorithm */
    typedef typename impl::forward<X, Y, emission_p_t> forward;

    /** Viterbi algorithm */
    typedef typename impl::viterbi<X, Y, emission_p_t> viterbi;

    /** Baum-Welch algorithm */
    typedef impl::baum_welch<X, Y, impl::hmm_emit_gaussian<Y>,
                             impl::baum_welch_emit_gaussian<X, Y> >
        baum_welch;

    /**
     *  \brief  Get emission probability (bell curve) parameters
     *
     *  \param  state   Model state
     *  \param  u       Mean
     *  \param  sigma2  Variance
     */
    inline void emission(state_t & state, Y & u, Y & sigma2) const {
        state.value.emit_p.get(u, sigma2);
    }

    /**
     *  \brief  Set emission probability (bell curve) parameters
     *
     *  \param  state   Model state
     *  \param  u       Mean
     *  \param  sigma2  Variance
     */
    inline void emission(state_t & state, const Y & u, const Y & sigma2) {
        state.value.emit_p.set(u, sigma2);
    }

    /**
     *  \brief  Deserialise model from input stream
     *
     *  Throws an exception on invalid model string.
     *
     *  \param  in  Inpiut stream
     */
    ghmm(std::istream & in) {
        if (!this->deserialise(in))
            throw std::runtime_error(
                "math::ghmm: deserialisation failed");
    }

    protected:

    /** Default constructor */
    ghmm():
        impl::hmm<X, Y, impl::hmm_emit_gaussian<Y> >()
    {}

};  // end of template class ghmm

}  // end of namespace math

#endif  // end of #ifndef math__hmm_hxx
