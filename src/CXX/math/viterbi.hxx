#ifndef math__viterbi_hxx
#define math__viterbi_hxx

/**
 *  \brief  Viterbi algorithm (for HMM)
 *
 *  See http://en.wikipedia.org/wiki/Viterbi_algorithm
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
#include "math/hmm_base.hxx"

#include <vector>
#include <cstdlib>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

namespace impl {

/**
 *  \brief  Viterbi algorithm for HMM
 *
 *  Viterbi algorithm implementation.
 *  Descendants will probably want to overload report member functions
 *  in the following 2 versions:
 *  \code
 *
 *  void report(
 *      size_t    time,
 *      const Y & y,
 *      state_t & state,
 *      real_t    probability);
 *  void report(
 *      size_t    time,
 *      const Y & y,
 *      state_t & origin,
 *      state_t & target,
 *      real_t    probability);
 *  \endcode
 *
 *  The 1st function shall be called on the very 1st observation item
 *  (so the \c time parameter shall be 0).
 *  For each \c state, it will provide info about the \c probability
 *  that the model is in that state after the 1st obs. item \c y.
 *
 *  The 2nd function shall be called for all other obs. items
 *  (at times > 0).
 *  For each \c target state, it will provide info about
 *  the \c probability that the model is in that state
 *  after the obs. item \c y, taking transition from \c origin state.
 *
 *  The default implementations are empty.
 *
 *  \tparam  X  Hidden random variable
 *  \tparam  Y  Observed random variable
 *  \tparam  P  Emission probability
 */
template <typename X, typename Y, typename P>
class viterbi {
    public:

    typedef hmm<X, Y, P>                   model_t;       /**< Model      */
    typedef typename model_t::state_t      state_t;       /**< State      */
    typedef typename model_t::transition_t transition_t;  /**< Transition */

    private:

    /** Hidden random variable value probability (old and new) */
    struct p_t { real_t p[2]; };

    /** Hidden random variables value probabilities */
    typedef std::vector<p_t> states_p_t;

    const model_t & m_model;     /**< Model       */
    states_p_t      m_states_p;  /**< States data */
    size_t          m_t;         /**< Time        */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  model  Model
     */
    viterbi(const model_t & model):
        m_model(model),
        m_states_p(model.state_cnt()),
        m_t(0)
    {}

    /**
     *  \brief  Reporting for time 0
     *
     *  Intended to be overloaded by user.
     *
     *  \param  time         Time (always 0)
     *  \param  y            Emission at time 0
     *  \param  state        State of the model
     *  \param  probability  Probability of \c state at time 0
     */
    virtual void report(
        size_t          time,
        const Y &       y,
        const state_t & state,
        const real_t &  probability)
    {}

    /**
     *  \brief  Reporting for time >0
     *
     *  Intended to be overloaded by user.
     *
     *  \param  time         Time (>0)
     *  \param  y            Emission at \c time
     *  \param  origin       State of the model
     *  \param  target       State of the model
     *  \param  probability  Probability of \c target state at \c time
     *                       (via transition from \c origin state)
     */
    virtual void report(
        size_t          time,
        const Y &       y,
        const state_t & origin,
        const state_t & target,
        const real_t &  probability)
    {}

    /**
     *  \brief  Perform one step of the algorithm
     *
     *  \param  y  Observation at the step time
     */
    void step(const Y & y) {
        if (0 == m_t) {
            // Initialise state probabilities
            m_model.for_each_state([&,this](const state_t & state) {
                real_t p = state.value.p * state.value.emit_p(y);

                m_states_p[state.value.index].p[0] = p;

                report(m_t, y, state, p);
            });
        }
        else {
            // Update state probabilities
            m_model.for_each_state([&,this](const state_t & state) {
                real_t          max_p  = 0.0;
                const state_t * argmax = NULL;

                m_model.for_each_trans_to(state,
                [&,this](const transition_t & trans) {
                    const state_t & from = trans.origin();
                    const state_t & to   = trans.target();

                    real_t p;
                    p  = m_states_p[from.value.index].p[(m_t + 1) % 2];
                    p *= trans.value.p;
                    p *= to.value.emit_p(y);

                    if (max_p <= p) {
                        max_p  = p;

                        argmax = &from;
                    }
                });

                m_states_p[state.value.index].p[m_t % 2] = max_p;

                report(m_t, y, *argmax, state, max_p);
            });
        }

        ++m_t;
    }

    /** Reset the algorithm (so that it would run from beginning) */
    inline void reset() { m_t = 0; }

};  // end of template class viterbi

}}  // end of namespaces impl and math

#endif  // end of #ifndef math__viterbi_hxx
