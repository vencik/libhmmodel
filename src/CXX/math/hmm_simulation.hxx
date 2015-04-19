#ifndef math__hmm_simulation_hxx
#define math__hmm_simulation_hxx

/**
 *  \brief  Simulation of random process modeled by HMM
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

#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

namespace impl {

/**
 *  \brief  Simulation of random process modeled by HMM
 *
 *  \tparam  X  Hidden random variable
 *  \tparam  Y  Observed random variable
 *  \tparam  P  Emission probability
 */
template <typename X, typename Y, typename P>
class hmm_simulation {
    public:

    typedef hmm<X, Y, P>                   model_t;       /**< Model      */
    typedef typename model_t::state_t      state_t;       /**< State      */
    typedef typename model_t::transition_t transition_t;  /**< Transition */

    private:

    static const size_t s_scale;  /** Probability precision factor */

    const model_t & m_model;  /**< Model         */
    const state_t * m_state;  /**< Current state */

    /** Probability-based state selection vector */
    std::vector<const state_t *> m_state_sel;

    /** Selector of initial state */
    class init_state_selector {
        private:

        const model_t & m_model;

        public:

        init_state_selector(const model_t & model): m_model(model) {}

        template <class I>
        void for_each(I inj) const {
            m_model.for_each_state([&](const state_t & state) {
                inj(&state, state.value.p);
            });
        }

    };  // end of class init_state_selector

    /** Selector of next state */
    class next_state_selector {
        private:

        const model_t & m_model;
        const state_t * m_state;

        public:

        next_state_selector(
            const model_t & model,
            const state_t * state)
        :
            m_model(model),
            m_state(state)
        {}

        template <class I>
        void for_each(I inj) const {
            m_model.for_each_trans_from(*m_state,
            [&](const transition_t & trans) {
                inj(&trans.target(), trans.value.p);
            });
        }

    };  // end of class next_state_selector

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  model  Model
     */
    hmm_simulation(const model_t & model):
        m_model(model),
        m_state(NULL),
        m_state_sel(s_scale)
    {}

    /**
     *  \brief  Perform one step of the algorithm
     *
     *  \return Emission
     */
    Y step() {
        m_state = NULL == m_state
            ? select_p(m_state_sel, init_state_selector(m_model))
            : select_p(m_state_sel, next_state_selector(m_model, m_state));

        assert(NULL != m_state);

        return m_state->value.emit_p.rand();
    }

    /** Reset simulation */
    void reset() { m_state = NULL; }

};  // end of template class hmm_simulation

// Initialisation of static members
template <typename X, typename Y, typename P>
const size_t hmm_simulation<X, Y, P>::s_scale = MATH_DEFAULT_RAND_SCALE;

}}  // end of namespaces impl and math

#endif  // end of #ifndef math__hmm_simulation_hxx
