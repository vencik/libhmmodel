#ifndef math__baum_welch_hxx
#define math__baum_welch_hxx

/**
 *  \brief  Baum-Welch algorithm
 *
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

#include "math/real.hxx"
#include "math/rng.hxx"
#include "math/hmm_base.hxx"
#include "math/hmm_discrete.hxx"
#include "math/hmm_gaussian.hxx"

#include <vector>
#include <list>
#include <cstdlib>
#include <cassert>
#include <cmath>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

namespace impl {

/**
 *  \brief  Baum-Welch algorithm for HMM
 *
 *  User may overload the \c report member function to get info
 *  about each step of the algorithm:
 *
 *  \code
 *    virtual void report(
 *        const model_t &                        model,
 *        const model_t::observation_t &         obs,
 *        const std::vector<math::real_vector> & alpha,
 *        const std::vector<math::real_vector> & beta,
 *        const std::vector<math::real_vector> & gamma,
 *        const std::vector<math::real_vector> & xi);
 *  \endcode
 *
 *  The template functor \c PI shall be instantiated like this:
 *
 *  \code
 *  PI pi(model);
 *  \endcode
 *
 *  and called thus:
 *
 *  \code
 *  pi(
 *      const std::list<std::vector<Y> > &                   obs,
 *      const std::vector<std::vector<math::real_vector> > & gamma,
 *      const std::vector<std::vector<math::real_vector> > & xi);
 *  \end code
 *
 *  and is responsible for implementation of emissions training.
 *
 *
 *  \tparam  X   Hidden random variable
 *  \tparam  Y   Observed random variable
 *  \tparam  P   Emission probability
 *  \tparam  PI  Emission probability setter
 */
template <typename X, typename Y, typename P, class PI>
class baum_welch {
    public:

    typedef hmm<X, Y, P>                    model_t;        /**< Model       */
    typedef typename model_t::state_t       state_t;        /**< Model state */
    typedef typename model_t::transition_t  transition_t;   /**< Model state */
    typedef typename model_t::observation_t observation_t;  /**< Model state */

    protected:

    model_t & m_model;  /**< Trained model */

    public:

    /**
     *  \brief Constructor
     *
     *  The constructor initialises the model parameters
     *  (states start probability and transitions probability)
     *  by random values if the \c init_pi and \c init_tr parameters
     *  are \c true, respectively.
     *  If at least one is \c true, the \c seed parameter may be used
     *  to specify the RNG seed
     *  (0 means that current time shall be used,
     *  -1 means not to seed the RNG, which is the default).
     *
     *  \param  model    Trained model
     *  \param  init_pi  Initialise states start probability
     *  \param  init_tr  Initialise transitions probability
     *  \param  init_e   Initialise emissions probability
     */
    baum_welch(
        model_t & model,
        bool      init_pi = false,
        bool      init_tr = false,
        bool      init_e  = false)
    :
        m_model(model)
    {
        if (!(init_pi || init_tr || init_e))
            return;  // no initialisation

        // Initialise states start probability
        if (init_pi) {
            auto pi = rand_p_vector(m_model.state_cnt());

            m_model.for_each_state([&pi](state_t & state) {
                state.value.p = pi[state.value.index];
            });
        }

        // Initialise transitions probability
        if (init_tr) {
            m_model.for_each_state([this](state_t & state) {
                auto A = rand_p_vector(m_model.transition_cnt_from(state));

                size_t ij = 0;
                m_model.for_each_trans_from(state,
                [&A,&ij](transition_t & trans) {
                    trans.value.p = A[ij++];
                });
            });
        }

        // Initialise emissions probability
        if (init_e) {
            m_model.for_each_state([this](state_t & state) {
                state.value.emit_p.set_rand();
            });
        }
    }

    /**
     *  \brief  Pre-update report
     *
     *  \param  model  Trained model (BEFORE the update)
     *  \param  obs    Training pattern
     *  \param  alpha  Forward procedure output
     *  \param  beta   Backward procedure output
     *  \param  gamma  Update output (for states)
     *  \param  xi     Update output (for transitions)
     */
    virtual void report(
        const model_t &                        model,
        const observation_t &                  obs,
        const std::vector<math::real_vector> & alpha,
        const std::vector<math::real_vector> & beta,
        const std::vector<math::real_vector> & gamma,
        const std::vector<math::real_vector> & xi)
    {}

    private:

    /**
     *  \brief  One step of the algorithm
     *
     *  The function uses externally-provided probability matrixes
     *  (pre-initialised to 0).
     *
     *  \param  obs    Observation
     *  \param  gamma  Prob. of being in a state at a time
     *  \param  xi     Prob. of transitions at a time
     */
    void step(
        const observation_t &            obs,
        std::vector<math::real_vector> & gamma,
        std::vector<math::real_vector> & xi)
    {
        size_t T = obs.size();  // training observation size

        assert(T > 0);

        // Forward procedure
        std::vector<math::real_vector> alpha;
        for (size_t i = 0; i < m_model.state_cnt(); ++i)
            alpha.emplace_back(T);

        m_model.for_each_state([&,this](state_t & i) {
            auto & alpha_i = alpha[i.value.index];

            alpha_i[0] = i.value.p * i.value.emit_p(obs[0]);
        });

        for (size_t t = 1; t < T; ++t)
            m_model.for_each_state([&,this](state_t & i) {
                auto & alpha_i = alpha[i.value.index];

                real_t sum_alpha_trans_p = 0.0;
                m_model.for_each_trans_to(i, [&](transition_t & trans) {
                    auto & j = trans.origin();
                    auto & alpha_j = alpha[j.value.index];

                    sum_alpha_trans_p += alpha_j[t - 1] * trans.value.p;
                });

                alpha_i[t] = i.value.emit_p(obs[t]) * sum_alpha_trans_p;
            });

        // Backward procedure
        std::vector<math::real_vector> beta;
        for (size_t i = 0; i < m_model.state_cnt(); ++i)
            beta.emplace_back(T);

        m_model.for_each_state([&,this](state_t & i) {
            auto & beta_i = beta[i.value.index];

            beta_i[T - 1] = 1.0;
        });

        for (size_t t_plus_1 = T - 1; t_plus_1; --t_plus_1) {
            size_t t = t_plus_1 - 1;

            m_model.for_each_state([&,this](state_t & i) {
                auto & beta_i = beta[i.value.index];

                real_t sum_beta_trans_emit = 0.0;
                m_model.for_each_trans_from(i,
                [&,this](transition_t & trans) {
                    auto & j = trans.target();
                    auto & beta_j = beta[j.value.index];

                    sum_beta_trans_emit +=
                        beta_j[t_plus_1] * trans.value.p *
                        j.value.emit_p(obs[t_plus_1]);
                });

                beta_i[t] = sum_beta_trans_emit;
            });
        }

        // Update
        math::real_vector sum_alpha_beta(T);

        for (size_t t = 0; t < T; ++t) {
            sum_alpha_beta[t] = 0.0;

            for (size_t i = 0; i < m_model.state_cnt(); ++i)
                sum_alpha_beta[t] += alpha[i][t] * beta[i][t];

            for (size_t i = 0; i < m_model.state_cnt(); ++i)
                gamma[i][t] =
                    alpha[i][t] * beta[i][t] / sum_alpha_beta[t];
        }

        for (size_t t = 0; t < T - 1; ++t) {
            size_t t_plus_1 = t + 1;

            m_model.for_each_trans([&,this](transition_t & trans) {
                auto & i = trans.origin();
                auto & j = trans.target();

                auto & alpha_i = alpha[i.value.index];
                auto & beta_j  = beta[j.value.index];

                real_t xi_ij_t;
                xi_ij_t  = alpha_i[t] * trans.value.p;
                xi_ij_t *= beta_j[t_plus_1];
                xi_ij_t *= j.value.emit_p(obs[t_plus_1]);
                xi_ij_t /= sum_alpha_beta[t];

                xi[trans.value.index][t] = xi_ij_t;
            });
        }

        report(m_model, obs, alpha, beta, gamma, xi);
    }

    public:

    /**
     *  \brief  Train model (using batch training)
     *
     *  The function perform one training batch.
     *  It provides back a measure of convergency.
     *
     *  \param  obs  Training observations (>1, must not be empty)
     *
     *  \return Sum of model parameters absolute differences
     */
    real_t train(const std::list<observation_t> & obs) {
        assert(!obs.empty());

        real_t e = 0.0;

        size_t D = obs.size();
        size_t M = m_model.state_cnt();
        size_t N = m_model.transition_cnt();

        // Initialise states & transitions probability
        std::vector<std::vector<math::real_vector> > gamma(D);
        std::vector<std::vector<math::real_vector> > xi(D);

        size_t d = 0;

        for (auto o = obs.begin(); o != obs.end(); ++o, ++d) {
            gamma[d].reserve(M);
            xi[d].reserve(N);

            for (size_t m = 0; m < M; ++m)
                gamma[d].emplace_back(o->size());

            for (size_t n = 0; n < N; ++n)
                xi[d].emplace_back(o->size() - 1);
        }

        d = 0;
        for (auto o = obs.begin(); o != obs.end(); ++o, ++d)
            step(*o, gamma[d], xi[d]);

        // Set start probability
        m_model.for_each_state([&,this](state_t & i) {
            real_t p_e = i.value.p;

            i.value.p = 0.0;
            for (d = 0; d < D; ++d)
                i.value.p += gamma[d][i.value.index][0];
            i.value.p /= D;

            p_e -= i.value.p;
            e   += p_e.abs();
        });

        // Set transitions probability
        m_model.for_each_trans([&,this](transition_t & ij) {
            real_t p_e = ij.value.p;

            auto & i = ij.origin();

            real_t sum_xi_ij   = 0.0;
            real_t sum_gamma_i = 0.0;

            for (d = 0; d < D; ++d) {
                auto & gamma_i = gamma[d][i.value.index];
                auto & xi_ij   = xi[d][ij.value.index];

                size_t T = gamma_i.rank();

                assert(T - 1 == xi_ij.rank());

                for (size_t t = 0; t < T - 1; ++t)
                    sum_xi_ij += xi_ij[t];

                for (size_t t = 0; t < T - 1; ++t)
                    sum_gamma_i += gamma_i[t];
            }

            ij.value.p = sum_xi_ij / sum_gamma_i;

            p_e -= ij.value.p;
            e   += p_e.abs();
        });

        // Set emission probability
        PI pi(m_model);
        e += pi(obs, gamma, xi);

        return e;
    }

};  // end of template class baum_welch


/** Baum-Welch training for categorial emissions */
template <typename X, typename Y>
class baum_welch_emit_categorial {
    public:

    typedef hmm<X, Y, hmm_emit_categorial<Y> > model_t;  /**< Model */

    typedef typename model_t::state_t       state_t;        /**< Model state */
    typedef typename model_t::transition_t  transition_t;   /**< Model state */
    typedef typename model_t::observation_t observation_t;  /**< Model state */

    private:

    model_t & m_model;  /**< Model */

    /**
     *  \brief  Emissions training (for 1 state)
     *
     *  \param  i          State index
     *  \param  obs        Training patterns
     *  \param  gamma      Prob. of being in a state at a time
     *  \param  xi         Prob. of transitions at a time
     *
     *  \return Sum of absolute differences of parameters
     */
    static real_t set_emission_p(
        state_t &                                            i,
        const std::list<std::vector<Y> > &                   obs,
        const std::vector<std::vector<math::real_vector> > & gamma,
        const std::vector<std::vector<math::real_vector> > & xi)
    {
        real_t e = 0.0;

        math::real_vector b_i(Y::cardinality());

        real_t sum_gamma_i = 0.0;
        real_t sum_b_i     = 0.0;

        size_t d = 0;
        for (auto o = obs.begin(); o != obs.end(); ++o, ++d) {
            auto & gamma_i = gamma[d][i.value.index];

            for (size_t k = 0; k < Y::cardinality(); ++k) {
                const Y & v_k = Y::value(k);

                real_t sum_chi_gamma_i = 0.0;
                for (size_t t = 0; t < o->size(); ++t)
                    if ((*o)[t] == v_k)
                        sum_chi_gamma_i += gamma_i[t];

                b_i[k] += sum_chi_gamma_i;
            }

            for (size_t t = 0; t < gamma_i.rank(); ++t)
                sum_gamma_i += gamma_i[t];
        }

        for (size_t k = 0; k < Y::cardinality(); ++k)
            sum_b_i += b_i[k] /= sum_gamma_i;

        // Normalise emission probabilities
        for (size_t k = 0; k < Y::cardinality(); ++k) {
            const Y & v_k = Y::value(k);
            real_t    p   = b_i[k] / sum_b_i;
            real_t    p_e = i.value.emit_p(v_k) - p;

            i.value.emit_p.set(v_k, p);

            e += p_e.abs();
        }

        return e;
    }

    public:

    /** Constructor */
    baum_welch_emit_categorial(model_t & model): m_model(model) {}

    /**
     *  \brief  Emission probabilities setter
     *
     *  \param  obs    Training patterns
     *  \param  gamma  Update output (for states)
     *  \param  xi     Update output (for transitions)
     *
     *  \return Sum of absolute differences of parameters
     */
    virtual real_t operator () (
        const std::list<std::vector<Y> > &                   obs,
        const std::vector<std::vector<math::real_vector> > & gamma,
        const std::vector<std::vector<math::real_vector> > & xi)
    {
        real_t e = 0.0;

        m_model.for_each_state([&,this](state_t & i) {
            e += set_emission_p(i, obs, gamma, xi);
        });

        return e;
    }

};  // end of template class baum_welch_emit_categorial


/** Baum-Welch training for N(u, sigma2) emissions */
template <typename X, typename Y>
class baum_welch_emit_gaussian {
    public:

    typedef hmm<X, Y, hmm_emit_gaussian<Y> > model_t;  /**< Model */

    typedef typename model_t::state_t       state_t;        /**< Model state */
    typedef typename model_t::transition_t  transition_t;   /**< Model state */
    typedef typename model_t::observation_t observation_t;  /**< Model state */

    private:

    model_t & m_model;  /**< Model */

    public:

    /** Constructor */
    baum_welch_emit_gaussian(model_t & model): m_model(model) {}

    /**
     *  \brief  Emission probabilities setter
     *
     *  \param  obs    Training patterns
     *  \param  gamma  Update output (for states)
     *  \param  xi     Update output (for transitions)
     *
     *  \return Sum of absolute differences of parameters
     */
    virtual real_t operator () (
        const std::list<std::vector<Y> > &                   obs,
        const std::vector<std::vector<math::real_vector> > & gamma,
        const std::vector<std::vector<math::real_vector> > & xi)
    {
        real_t e = 0.0;

        assert(!obs.empty());

        m_model.for_each_state([&,this](state_t & i) {
            Y u_i, sigma2_i;

            real_t f = 0.0;

            // Estimate u
            auto obs_k_iter = obs.begin();
            for (size_t k = 0; obs_k_iter != obs.end(); ++obs_k_iter, ++k) {
                const auto & obs_k = *obs_k_iter;

                for (size_t t = 0; t < obs_k.size(); ++t) {
                    const auto & obs_k_t = obs_k[t];

                    m_model.for_each_trans_from(i,
                    [&,this](const transition_t & ij) {
                        const auto & xi_ij_k = xi[k][ij.value.index];

                        for (size_t l = 0; l < u_i.rank(); ++l)
                            u_i[l] += xi_ij_k[t] * obs_k_t[l];

                        f += xi_ij_k[t];
                    });
                }
            }

            for (size_t l = 0; l < u_i.rank(); ++l) {
                u_i[l] /= f;

                real_t p_e = i.value.emit_p.u()[l] - u_i[l];
                e += p_e.abs();
            }

            // Estimate sigma^2
            obs_k_iter = obs.begin();
            for (size_t k = 0; obs_k_iter != obs.end(); ++obs_k_iter, ++k) {
                const auto & obs_k = *obs_k_iter;

                for (size_t t = 0; t < obs_k.size(); ++t) {
                    const auto & obs_k_t = obs_k[t];

                    m_model.for_each_trans_from(i,
                    [&,this](const transition_t & ij) {
                        const auto & xi_ij_k = xi[k][ij.value.index];

                        for (size_t l = 0; l < u_i.rank(); ++l) {
                            real_t delta = u_i[l] - obs_k_t[l];

                            sigma2_i[l] += xi_ij_k[t] * delta * delta;
                        }
                    });
                }
            }

            for (size_t l = 0; l < sigma2_i.rank(); ++l) {
                sigma2_i[l] /= f;

                real_t p_e = i.value.emit_p.sigma2()[l] - sigma2_i[l];
                e += p_e.abs();
            }

            i.value.emit_p.set(u_i, sigma2_i);
        });

        return e;
    }

};  // end of template class baum_welch_emit_gaussian

}}  // end of namespaces impl and math

#endif  // end of #ifndef math__baum_welch_hxx
