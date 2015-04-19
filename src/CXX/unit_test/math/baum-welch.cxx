/**
 *  See http://en.wikipedia.org/wiki/Baum-Welch_algorithm
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

#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cassert>
#include <list>
#include <vector>
#include <cstdarg>
#include <cstdlib>
#include <ctime>

#ifndef HAVE_CXX11
#error "Sorry, C++11 support is required to compile this"
#endif


/** Hidden random variable */
enum state_t {
    STATE1 = 0,
    STATE2
};  // end of enum state_t

/** Translate state_t to string */
static const char * const state_t_str[] = {
    /* STATE1 */  "State 1",
    /* STATE2 */  "State 2",
};  // end of state_t_str array

/** State serialisation */
inline static std::ostream & operator << (
    std::ostream & out, state_t state)
{
    return out << state_t_str[state];
}


/** Output random variable */
enum emission_value_t {
    NO_EGGS = 0,
    EGGS,
};  // end of enum emission_value_t

static const char * const emission_value_t_str[] = {
    /* NO_EGGS */  "No eggs",
    /* EGGS    */  "Eggs",
};  // end of emission_value_t_str array

/** Emission serialisation */
std::ostream & operator << (
    std::ostream & out, const emission_value_t & e)
{
    return out << emission_value_t_str[e];
}

/** Output random variable (wrapper) */
class emission_t: public math::hmm_emission_t<emission_value_t> {
    public:

    /** Constructor */
    emission_t(emission_value_t v):
        math::hmm_emission_t<emission_value_t>(v)
    {}

    static size_t cardinality() { return 2; }

    static emission_t value(size_t index) {
        return emission_t((emission_value_t)index);
    }

};  // end of class emission_t


/** Model implementation */
typedef math::hmm<state_t, emission_t> model_impl_t;

/** Model */
class model: public model_impl_t {
    public:

    /** Construct the model */
    model() {
        // States & their start probabilities
        model_impl_t::state_t & state1 = state(STATE1, 0.5);
        model_impl_t::state_t & state2 = state(STATE2, 0.5);

        // Transition table
        transition(state1, state1, 0.5);
        transition(state1, state2, 0.5);
        transition(state2, state1, 0.3);
        transition(state2, state2, 0.7);

        // Emission table
        emission(state1, NO_EGGS, 0.3);
        emission(state1, EGGS,    0.7);
        emission(state2, NO_EGGS, 0.8);
        emission(state2, EGGS,    0.2);
    }

    /** Baum-Welch algorithm (debugging reports) */
    class baum_welch: public model_impl_t::baum_welch {
        public:

        /** Model type */
        typedef model_impl_t::baum_welch::model_t model_t;

        private:

        /**
         *  \brief  Report observation
         *
         *  \brief  label  Label
         *  \brief  obs    Observation
         */
        void report_obs(
            const std::string &            label,
            const model_t::observation_t & obs) const
        {
            std::cerr << label << ':' << std::endl;

            for (auto i = obs.begin(); i != obs.end(); ++i)
                std::cerr << " (" << *i << ")";

            std::cerr << std::endl;
        }

        /**
         *  \brief  Report probabilities table
         *
         *  \brief  label  Label
         *  \brief  table  Table
         */
        void report_table(
            const std::string &                       label,
            const std::vector<std::vector<double> > & table) const
        {
            std::cerr << label << ':' << std::endl;

            for (size_t i = 0; i < table.size(); ++i) {
                std::cerr << i << ':';

                for (auto t = table[i].begin(); t != table[i].end(); ++t)
                    std::cerr << ' ' << *t;

                std::cerr << std::endl;
            }
        }

        public:

        /** Constructor */
        baum_welch(
            model_t & model,
            bool      init_tr = false,
            bool      init_pi = false,
            int       seed    = -1)
        :
            model_impl_t::baum_welch(model, init_tr, init_pi, seed)
        {}

        /**
         *  \brief  Debug pre-update report
         *
         *  \param  model  Trained model
         *  \param  obs    Training pattern
         *  \param  alpha  Forward procedure output
         *  \param  beta   Backward procedure output
         *  \param  gamma  Update output (for states)
         *  \param  xi     Update output (for transitions)
         */
        void report(
            const model_t &                           model,
            const model_t::observation_t &            obs,
            const std::vector<std::vector<double> > & alpha,
            const std::vector<std::vector<double> > & beta,
            const std::vector<std::vector<double> > & gamma,
            const std::vector<std::vector<double> > & xi)
        {
            report_obs("Training pattern", obs);

            std::cerr << std::endl;

            report_table("alpha", alpha);

            std::cerr << std::endl;

            report_table("beta", beta);

            std::cerr << std::endl;

            report_table("gamma", gamma);

            std::cerr << std::endl;

            report_table("xi", xi);

            std::cerr << std::endl;
        }

    };  // end of class baum_welch

};  // end of class model


/** Create observation */
static std::vector<emission_t> observation(size_t T...) {
    std::vector<emission_t> obs;
    obs.reserve(T);

    va_list args;
    va_start(args, T);

    for (size_t i = 0; i < T; ++i) {
        obs.push_back((emission_value_t)va_arg(args, int));
    }

    va_end(args);

    return obs;
}


/** Compare observations (lexicographic comparison) */
bool operator < (
    const std::vector<emission_t> v1,
    const std::vector<emission_t> v2)
{
    for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
        if (v1[i] < v2[i]) return true;
        if (v2[i] < v1[i]) return false;
    }

    return v1.size() < v2.size();
}


/** Serialise observation */
std::ostream & operator << (
    std::ostream & out, const std::vector<emission_t> & obs)
{
    out << "observation:";

    for (auto i = obs.begin(); i != obs.end(); ++i)
        out << " \"" << *i << '"';

    return out;
}


/**
 *  \brief  Baum-Welch algorithm unit test
 *
 *  \param  training_loops  Number of training batches
 *                          (0 means as many as required to converge
 *                          below error of \c convergency_e)
 *  \param  testing_loops   Number of generated observations
 *  \param  convergency_e   Convergency error threshold
 *
 *  \return Exit code
 */
static int baum_welch_test(
    size_t training_loops,
    size_t testing_loops,
    double convergency_e)
{
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        static const auto NN = observation(2, NO_EGGS, NO_EGGS);
        static const auto NE = observation(2, NO_EGGS, EGGS);
        static const auto EN = observation(2, EGGS,    NO_EGGS);
        static const auto EE = observation(2, EGGS   , EGGS);

        model m;

        // Train model
        std::list<std::vector<emission_t> > training_set;
        training_set.push_back(NN);
        training_set.push_back(NN);
        training_set.push_back(NN);
        training_set.push_back(NN);
        training_set.push_back(NE);
        training_set.push_back(EE);
        training_set.push_back(EE);
        training_set.push_back(EN);
        training_set.push_back(NN);
        training_set.push_back(NN);

        model::baum_welch bw(m, true, true, true);

        m.serialise("Chicken (initial)", std::cout);

        double e;

        if (0 < training_loops) {
            for (size_t l = 0; l < training_loops; ++l) {
                e = bw.train(training_set);
            }
        }
        else {
            training_loops = 0;

            do {
                e = bw.train(training_set);

                ++training_loops;

            } while (e > convergency_e);
        }

        m.serialise("Chicken", std::cout);

        std::cout
            << "Convergency error of " << e
            << " reached in " << training_loops
            << " loops" << std::endl;

        // Test empyrical probabilities of observations
        std::map<std::vector<emission_t>, unsigned> obs_cnt;
        obs_cnt[NN] = 0;
        obs_cnt[NE] = 0;
        obs_cnt[EN] = 0;
        obs_cnt[EE] = 0;

        for (size_t i = 0; i < testing_loops; ++i) {
            model::simulation s(m);

            std::vector<emission_t> obs;
            obs.reserve(2);
            for (size_t t = 0; t < 2; ++t)
                obs.push_back(s.step());

            ++obs_cnt[obs];
        }

        for (auto i = obs_cnt.begin(); i != obs_cnt.end(); ++i) {
            std::cout
                << i->first << ", P = "
                << (double)(i->second) / testing_loops
                << std::endl;
        }

        exit_code = 0;  // success

    } while (0);  // end of pragmatic loop

    return exit_code;
}

/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    // Initialise RNG
    unsigned rng_seed;
    if (argc > 4)
        rng_seed = ::atoi(argv[4]);
    else
        rng_seed = ::time(NULL);

    std::cerr
        << "RNG seed: " << rng_seed << std::endl;

    ::srand(rng_seed);

    size_t training_loops = 0;
    size_t testing_loops  = 1000;
    double convergency_e  = 0.0000001;

    if (argc > 1)
        training_loops = ::atoi(argv[1]);

    if (argc > 2)
        testing_loops = ::atoi(argv[2]);

    if (argc > 3)
        convergency_e = ::atof(argv[3]);

    int exit_code = baum_welch_test(
        training_loops, testing_loops, convergency_e);

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
