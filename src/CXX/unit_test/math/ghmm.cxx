/**
 *  Unit test for Hidden Markov model with Gaussian emissions
 *
 *  \date    2015/05/13
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
    S0 = 0,
    S1,
    S2,
    S3,
};  // end of enum state_t

/** Translate state_t to string */
static const char * const state_t_str[] = {
    /* S0 */  "S0",
    /* S1 */  "S1",
    /* S2 */  "S2",
    /* S3 */  "S3",
};  // end of state_t_str array

/** State serialisation */
std::ostream & operator << (std::ostream & out, state_t state) {
    return out << state_t_str[state];
}


/** Output N-dimensional random variable */
template <size_t N>
class emission_vector: public math::real_vector {
    public:

    /** Rank */
    static size_t rank() { return N; }

    /** Default constructor */
    emission_vector(): math::real_vector(N) {}

    /**
     *  \brief  Constructor
     *
     *  \param  i  Initialiser
     */
    emission_vector(const math::real_t & i): math::real_vector(N, i) {}

    /**
     *  \brief  Constructor
     *
     *  \param  il  Initialiser list
     */
    emission_vector(const std::initializer_list<math::real_t> & il):
        math::real_vector(il) {}

};  // end of class emission_vector

/** Output 2D random variable */
typedef emission_vector<2> emission_t;


/** Model implementation */
typedef math::ghmm<state_t, emission_t> model_impl_t;

/** Model */
class model: public model_impl_t {
    public:

    /** Construct the model */
    model() {
        // States & their start probabilities
        model_impl_t::state_t & s0 = state(S0, 1.0);
        model_impl_t::state_t & s1 = state(S1, 0.0);
        model_impl_t::state_t & s2 = state(S2, 0.0);
        model_impl_t::state_t & s3 = state(S3, 0.0);

        // Transitions
        transition(s0, s0, 0.5);
        transition(s0, s1, 0.5);

        transition(s1, s1, 0.7);
        transition(s1, s2, 0.3);

        transition(s2, s2, 0.8);
        transition(s2, s3, 0.2);

        transition(s3, s3, 1.0);

        // Emission parameters (mean & sigma^2)
        emission(s0, {1.0, 0.0}, {1.0, 1.0});
        emission(s1, {0.1, 1.1}, {0.9, 0.9});
        emission(s2, {1.2, 0.2}, {0.8, 0.8});
        emission(s3, {0.3, 1.3}, {0.7, 0.7});
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
        const math::real_t   & p)
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
            << "\tstate:       "  << state.value.x << std::endl
            << "\temission:    [" << y[0] << ' ' << y[1] << ']' << std::endl
            << "\tprobability: "  << p << std::endl;
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
            << "\ttarget:      " << target.value.x << std::endl
            << "\temission:    [" << y[0] << ' ' << y[1] << ']' << std::endl
            << "\torigin:      " << origin.value.x << std::endl
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
            std::cout << " " << (*i)->value.x;
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

        m.serialise("Gaussian model", std::cout);

        viterbi v(m);

        v.step({-0.1,  1.3});
        v.step({-0.1,  1.3});
        v.step({ 0.9, -0.2});
        v.step({ 0.0,  1.0});
        v.step({-0.2,  1.0});
        //v.step({ 0.9,  0.0});
        v.step({ 0.5, -0.3});
        //v.step({ 0.8, -0.1});
        v.step({ 0.2, -0.1});
        v.step({-0.1,  1.1});

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
            << std::endl;;
    }
    catch (...) {
        std::cerr
            << "Unhandled non-standard exception caught"
            << std::endl;
    }

    return exit_code;
}
