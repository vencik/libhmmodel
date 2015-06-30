/**
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

#include "math/hmm.hxx"
#include "math/numerics.hxx"

#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cassert>
#include <list>

#ifndef HAVE_CXX11
#error "Sorry, C++11 support is required to compile this"
#endif


/** Hidden random variable */
enum state_t {
    HEALTHY = 0,
    FEVER
};  // end of enum state_t

/** Translate state_t to string */
static const char * const state_t_str[] = {
    /* HEALTHY */  "Healthy",
    /* FEVER   */  "Fever",
};  // end of state_t_str array

/** Output random variable */
enum emission_value_t {
    DIZZY = 0,
    COLD,
    NORMAL
};  // end of enum emission_value_t

/** Output random variable (wrapper) */
class emission_t {
    private:

    const emission_value_t m_value;

    public:

    emission_t(emission_value_t value): m_value(value) {}

    operator emission_value_t () const { return m_value; }

    static size_t cardinality() { return 3; }

    static emission_t value(size_t index) {
        return emission_t((emission_value_t)index);
    }

};  // end of class emission_t

/** Translate emission_t to string */
static const char * const emission_t_str[] = {
    /* DIZZY  */  "Dizzy",
    /* COLD   */  "Cold",
    /* NORMAL */  "Normal",
};  // end of emission_t_str array


/** Model implementation */
typedef math::hmm<state_t, emission_t> model_impl_t;

/** Model */
class model: public model_impl_t {
    public:

    /** Construct the model */
    model() {
        // States & their start probabilities
        model_impl_t::state_t & healthy = state(HEALTHY, 0.6);
        model_impl_t::state_t & fever   = state(FEVER,   0.4);

        // Transitions
        transition(healthy, healthy, 0.7);
        transition(healthy, fever,   0.3);

        transition(fever, fever,   0.6);
        transition(fever, healthy, 0.4);

        // Emission probabilities
        emission(healthy, DIZZY,  0.1);
        emission(healthy, COLD,   0.4);
        emission(healthy, NORMAL, 0.5);

        emission(fever, DIZZY,  0.6);
        emission(fever, COLD,   0.3);
        emission(fever, NORMAL, 0.1);
    }

};  // end of class model


/** Viterbi algorithm */
class viterbi: public model_impl_t::viterbi {
    private:

    /** Hidden variable sequence (or "path") */
    typedef std::list<const model::state_t *> path_t;

    const model::state_t * m_state;  /**< Most probable actual state       */
    path_t                 m_path;   /**< Most probable path to \c m_state */
    math::real_t           m_p;      /**< Probability of the path          */

    /**
     *  \brief  Update path (by most probable transition)
     *
     *  \param  t       Time (>0)
     *  \param  origin  State of origin
     *  \param  target  Target state
     *  \param  p       Probability
     */
    void update(
        size_t                 t,
        const model::state_t & origin,
        const model::state_t & target,
        const math::real_t &   p)
    {
        --t;

        // Select maximum
        if (t < m_path.size()) {
            if (m_p <= p) {
                m_p  = p;
                m_path.back() = &origin;
                m_state = &target;
            }
        }

        // New item
        else {
            m_p = p;
            m_path.push_back(&origin);
            m_state = &target;
        }
    }

    public:

    viterbi(const model & m):
        model_impl_t::viterbi(m),
        m_p(0.0)
    {}

    void report(
        size_t                 t0,
        const emission_t &     y,
        const model::state_t & state,
        const math::real_t &   p)
    {
        assert(0 == t0);

        std::cerr
            << "Time: " << t0 << std::endl
            << "\tstate:       " << state_t_str[state.value.x] << std::endl
            << "\temission:    " << emission_t_str[y] << std::endl
            << "\tprobability: " << p << std::endl;
    }

    void report(
        size_t                 t,
        const emission_t &     y,
        const model::state_t & origin,
        const model::state_t & target,
        const math::real_t &   p)
    {
        assert(t > 0);

        std::cerr
            << "Time: " << t << std::endl
            << "\ttarget:      " << state_t_str[target.value.x] << std::endl
            << "\temission:    " << emission_t_str[y] << std::endl
            << "\torigin:      " << state_t_str[origin.value.x] << std::endl
            << "\tprobability: " << p << std::endl;

        update(t, origin, target, p);
    }

    /**
     *  \brief  Report path
     */
    void report() {
        m_path.push_back(m_state);

        std::cout
            << "Path (probability " << m_p << "):";

        for (auto i = m_path.begin(); i != m_path.end(); ++i) {
            std::cout << " " << state_t_str[(*i)->value.x];
        }

        std::cout
            << std::endl;
    }

};  // end of class viterbi


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        model m;

        viterbi v(m);

        v.step(NORMAL);
        v.step(COLD);
        v.step(DIZZY);

        v.report();

        exit_code = 0;  // success

    } while (0);  // end of pragmatic loop

    std::cerr
        << "Exit code: " << exit_code
        << std::endl;

    return exit_code;
}

/** Unit test exception-safe wrapper */
int main(int argc, char * const argv[]) {
    int exit_code = 128;

    try {
        exit_code = main_impl(argc, argv);
    }
    catch (const std::exception & x) {
        std::cerr
            << "Standard exception caught: "
            << x.what()
            << std::endl;
    }
    catch (...) {
        std::cerr
            << "Unhandled non-standard exception caught"
            << std::endl;
    }

    return exit_code;
}
