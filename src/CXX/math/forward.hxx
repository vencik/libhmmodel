#ifndef math__forward_hxx
#define math__forward_hxx

/**
 *  \brief  Forward algorithm (for HMM)
 *
 *  See http://en.wikipedia.org/wiki/Forward_algorithm
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

#include "math/real.hxx"
#include "math/hmm_base.hxx"

#include <vector>
#include <cstdlib>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

namespace impl {

/**
 *  \brief  Forward algorithm for HMM
 *
 *  \tparam  X  Hidden random variable
 *  \tparam  Y  Observed random variable
 *  \tparam  P  Emission probability
 */
template <typename X, typename Y, typename P>
class forward {
    public:

    typedef hmm<X, Y, P>                   model_t;       /**< Model      */
    typedef typename model_t::state_t      state_t;       /**< State      */
    typedef typename model_t::transition_t transition_t;  /**< Transition */

    private:

    /** Probability of being in a state at times t and t - 1 */
    struct p_t { real_t p[2]; };

    const model_t &  m_model;  /**< Model                          */
    std::vector<p_t> m_alpha;  /**< Probability of being in states */
    size_t           m_i;      /**< Index to \c m_alpha for time t */

    /** alpha_t(x) (for 0 <= t <= 1) */
    inline real_t & alpha(size_t t, const state_t & state) const {
        return m_alpha[state.value.index].p[t];
    }

    public:

    /** Initialise state probabilities */
    void reset() {
        m_model.for_each_state([this](const state_t & state) {
            alpha_t(1, state) = state.value.p;
        });

        m_i = 1;  // -1 actually, we're indexing from 0 as usual
    }

    /**
     *  \brief  Constructor
     *
     *  \param  model  Model
     */
    forward(const model_t & model):
        m_model(model),
        m_alpha(m_model.state_cnt())
    {
        reset();
    }

    /**
     *  \brief  Perform one step of the algorithm
     *
     *  \param  y  Observation at the step time
     */
    void step(const Y & y) {
        size_t m_i_prev = m_i;
        m_i = (m_i + 1) % 2;

        m_model.for_each_state([&,this](const state_t & state) {
            real_t sum_p = 0.0;
            m_model.for_each_trans_to(state,
            [&,this](const transition_t & trans) {
                sum_p += trans.value.p * m_alpha(m_i_prev, trans.origin());
            });

            m_alpha(m_i, state) = state.value.emit_p(y) * sum_p;
        });
    }

    /** State probability getter */
    inline real_t operator () (const state_t & state) const {
        return m_alpha(m_i, state);
    }

};  // end of template class forward

}}  // end of namespaces impl and math

#endif  // end of #ifndef math__forward_hxx
