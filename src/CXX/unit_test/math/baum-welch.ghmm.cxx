/**
 *  \brief  Baum--Welch algorithm for HMM with Gaussian emissions
 *
 *  See http://en.wikipedia.org/wiki/Baum-Welch_algorithm
 *
 *  \date    2015/05/22
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
#include <initializer_list>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime>

#ifndef HAVE_CXX11
#error "Sorry, C++11 support is required to compile this"
#endif


/** Hidden random variable */
enum state_t {
    S0 = 0,
    S1,
    S2,
    S3
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
    emission_vector(
        const std::initializer_list<math::real_vector::base_t> & il)
    :
        math::real_vector(il)
    {}

    /** Comparison */
    bool operator < (const emission_vector & rarg) const {
        for (size_t d = 0; d < rank(); ++d)
            if ((*this)[d] != rarg[d]) return (*this)[d] < rarg[d];

        return false;
    }

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
/*
        emission(s0, {1.0, 0.0}, {1.0, 1.0});
        emission(s1, {0.1, 1.1}, {0.9, 0.9});
        emission(s2, {1.2, 0.2}, {0.8, 0.8});
        emission(s3, {0.3, 1.3}, {0.7, 0.7});
*/
        emission(s0, {0.0, 0.0}, {1.0, 1.0});
        emission(s1, {0.0, 0.0}, {1.0, 1.0});
        emission(s2, {0.0, 0.0}, {1.0, 1.0});
        emission(s3, {0.0, 0.0}, {1.0, 1.0});
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

            std::for_each(obs.begin(), obs.end(),
            [](const emission_t & e) {
                std::cerr << " [" << e[0] << ' ' << e[1] << ']';
            });

            std::cerr << std::endl;
        }

        /**
         *  \brief  Report probabilities table
         *
         *  \brief  label  Label
         *  \brief  table  Table
         */
        void report_table(
            const std::string &                    label,
            const std::vector<math::real_vector> & table) const
        {
            std::cerr << label << ':' << std::endl;

            for (size_t i = 0; i < table.size(); ++i) {
                std::cerr << i << ':';

                for (size_t t = 0; t < table[i].rank(); ++t)
                    std::cerr << ' ' << table[i][t];

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
            const model_t &                        model,
            const model_t::observation_t &         obs,
            const std::vector<math::real_vector> & alpha,
            const std::vector<math::real_vector> & beta,
            const std::vector<math::real_vector> & gamma,
            const std::vector<math::real_vector> & xi)
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
static std::vector<emission_t> observation(
    const std::initializer_list<
        const std::initializer_list<math::real_vector::base_t> > & emissions)
{
    std::vector<emission_t> obs;
    obs.reserve(emissions.size());

    std::for_each(emissions.begin(), emissions.end(),
    [&obs](const std::initializer_list<math::real_vector::base_t> & emission) {
        obs.push_back(emission);
    });

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

    std::for_each(obs.begin(), obs.end(),
    [&out](const emission_t & emission) {
        out << " \"" << emission << '"';
    });

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
        // Train model
        std::list<std::vector<emission_t> > training_set;

        training_set.push_back(
            observation({
                {1.0, 0.0},
                {0.5, 0.5},
                //{1.1,-0.1},
                {1.0, 0.0},
                {0.4,-0.6},
                //{0.5,-0.5},
                //{0.4,-0.6},
            }));

        training_set.push_back(
            observation({
                {0.99, 0.01},
                {0.5,  0.51},
                //{1.1, 0.0},
                {1.0,  0.0 },
                {0.4, -0.58},
                //{0.49,-0.51},
                //{0.5,-0.5},
            }));

        model m;
        model::baum_welch bw(m, true, true, true);

        m.serialise("Gaussian HMM", std::cout);

        math::real_t e;

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

            } while (e > convergency_e && 999 > training_loops);
        }

        m.serialise("Test", std::cout);

        std::cout
            << "Convergency error of " << e
            << " reached in " << training_loops
            << " loops" << std::endl;

#if (0)
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
#endif

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
