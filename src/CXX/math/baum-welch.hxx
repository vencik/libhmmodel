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

#include "math/hmm_base.hxx"
#include "math/hmm_discrete.hxx"
#include "math/hmm_gaussian.hxx"
#include "math/rng.hxx"

#include <vector>
#include <list>
#include <cstdlib>
#include <cassert>
#include <cmath>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


/** Absolute value */
#define abs(x) ((x) < 0 ? -(x) : (x))


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
 *        const model_t &                           model,
 *        const model_t::observation_t &            obs,
 *        const std::vector<std::vector<double> > & alpha,
 *        const std::vector<std::vector<double> > & beta,
 *        const std::vector<std::vector<double> > & gamma,
 *        const std::vector<std::vector<double> > & xi);
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
 *      const std::list<std::vector<Y> > &                      obs,
 *      const std::vector<std::vector<std::vector<double> > > & gamma,
 *      const std::vector<std::vector<std::vector<double> > > & xi);
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
            auto pi = rand_p(m_model.state_cnt());

            m_model.for_each_state([&pi](state_t & state) {
                state.value.p = pi[state.value.index];
            });
        }

        // Initialise transitions probability
        if (init_tr) {
            m_model.for_each_state([this](state_t & state) {
                auto A = rand_p(m_model.transition_cnt_from(state));

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
        const model_t &                           model,
        const observation_t &                     obs,
        const std::vector<std::vector<double> > & alpha,
        const std::vector<std::vector<double> > & beta,
        const std::vector<std::vector<double> > & gamma,
        const std::vector<std::vector<double> > & xi)
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
        const observation_t &               obs,
        std::vector<std::vector<double> > & gamma,
        std::vector<std::vector<double> > & xi)
    {
        size_t T = obs.size();  // training observation size

        assert(T > 0);

        // Forward procedure
        std::vector<std::vector<double> > alpha(m_model.state_cnt());

        m_model.for_each_state([&,this](state_t & i) {
            auto & alpha_i = alpha[i.value.index];
            alpha_i.assign(T, 0.0);

            alpha_i[0] = i.value.p * i.value.emit_p(obs[0]);
        });

        for (size_t t = 1; t < T; ++t)
            m_model.for_each_state([&,this](state_t & i) {
                auto & alpha_i = alpha[i.value.index];

                double sum_alpha_trans_p = 0.0;
                m_model.for_each_trans_to(i, [&](transition_t & trans) {
                    auto & j = trans.origin();
                    auto & alpha_j = alpha[j.value.index];

                    sum_alpha_trans_p += alpha_j[t - 1] * trans.value.p;
                });

                alpha_i[t] = i.value.emit_p(obs[t]) * sum_alpha_trans_p;
            });

        // Backward procedure
        std::vector<std::vector<double> > beta(m_model.state_cnt());

        m_model.for_each_state([&,this](state_t & i) {
            auto & beta_i = beta[i.value.index];
            beta_i.assign(T, 0.0);

            beta_i[T - 1] = 1.0;
        });

        for (size_t t_plus_1 = T - 1; t_plus_1; --t_plus_1) {
            size_t t = t_plus_1 - 1;

            m_model.for_each_state([&,this](state_t & i) {
                auto & beta_i = beta[i.value.index];

                double sum_beta_trans_emit = 0.0;
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
        std::vector<double> sum_alpha_beta(T);

        for (size_t i = 0; i < m_model.state_cnt(); ++i)
            gamma[i].assign(T, 0.0);

        for (size_t t = 0; t < T; ++t) {
            sum_alpha_beta[t] = 0.0;

            for (size_t i = 0; i < m_model.state_cnt(); ++i)
                sum_alpha_beta[t] += alpha[i][t] * beta[i][t];

            for (size_t i = 0; i < m_model.state_cnt(); ++i)
                gamma[i][t] =
                    alpha[i][t] * beta[i][t] / sum_alpha_beta[t];
        }

        for (size_t ij = 0; ij < m_model.transition_cnt(); ++ij)
            xi[ij].assign(T - 1, 0.0);

        for (size_t t = 0; t < T - 1; ++t) {
            size_t t_plus_1 = t + 1;

            m_model.for_each_trans([&,this](transition_t & trans) {
                auto & i = trans.origin();
                auto & j = trans.target();

                auto & alpha_i = alpha[i.value.index];
                auto & beta_j  = beta[j.value.index];

                double xi_ij_t;
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
    double train(const std::list<observation_t> & obs) {
        assert(!obs.empty());

        double e = 0.0;

        size_t D = obs.size();
        size_t M = m_model.state_cnt();
        size_t N = m_model.transition_cnt();

        // Initialise states probability
        std::vector<std::vector<std::vector<double> > > gamma(D);

        for (auto i = gamma.begin(); i != gamma.end(); ++i) {
            i->reserve(M);

            for (size_t m = 0; m < M; ++m)
                i->push_back(std::vector<double>());
        }

        // Initialise transitions probability
        std::vector<std::vector<std::vector<double> > > xi(D);

        for (auto i = xi.begin(); i != xi.end(); ++i) {
            i->reserve(N);

            for (size_t n = 0; n < N; ++n)
                i->push_back(std::vector<double>());
        }

        size_t d = 0;
        for (auto o = obs.begin(); o != obs.end(); ++o, ++d)
            step(*o, gamma[d], xi[d]);

        // Set start probability
        m_model.for_each_state([&,this](state_t & i) {
            double p_e = i.value.p;

            i.value.p = 0.0;
            for (d = 0; d < D; ++d)
                i.value.p += gamma[d][i.value.index][0];
            i.value.p /= D;

            p_e -= i.value.p;
            e   += abs(p_e);
        });

        // Set transitions probability
        m_model.for_each_trans([&,this](transition_t & ij) {
            double p_e = ij.value.p;

            auto & i = ij.origin();

            double sum_xi_ij   = 0.0;
            double sum_gamma_i = 0.0;

            for (d = 0; d < D; ++d) {
                auto & gamma_i = gamma[d][i.value.index];
                auto & xi_ij   = xi[d][ij.value.index];

                size_t T = gamma_i.size();

                assert(T - 1 == xi_ij.size());

                for (auto t = xi_ij.begin(); t != xi_ij.end(); ++t)
                    sum_xi_ij += *t;

                for (size_t t = 0; t < T - 1; ++t)
                    sum_gamma_i += gamma_i[t];
            }

            ij.value.p = sum_xi_ij / sum_gamma_i;

            p_e -= ij.value.p;
            e   += abs(p_e);
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
    static double set_emission_p(
        state_t &                                               i,
        const std::list<std::vector<Y> > &                      obs,
        const std::vector<std::vector<std::vector<double> > > & gamma,
        const std::vector<std::vector<std::vector<double> > > & xi)
    {
        double e = 0.0;

        std::vector<double> b_i(Y::cardinality(), 0.0);
        double sum_gamma_i = 0.0;
        double sum_b_i     = 0.0;

        size_t d = 0;
        for (auto o = obs.begin(); o != obs.end(); ++o, ++d) {
            auto & gamma_i = gamma[d][i.value.index];

            for (size_t k = 0; k < Y::cardinality(); ++k) {
                const Y & v_k = Y::value(k);

                double sum_chi_gamma_i = 0.0;
                for (size_t t = 0; t < o->size(); ++t)
                    if ((*o)[t] == v_k)
                        sum_chi_gamma_i += gamma_i[t];

                b_i[k] += sum_chi_gamma_i;
            }

            auto gamma_i_t = gamma_i.begin();
            for (; gamma_i_t != gamma_i.end(); ++gamma_i_t)
                sum_gamma_i += *gamma_i_t;
        }

        for (size_t k = 0; k < Y::cardinality(); ++k)
            sum_b_i += b_i[k] /= sum_gamma_i;

        // Normalise emission probabilities
        for (size_t k = 0; k < Y::cardinality(); ++k) {
            const Y & v_k = Y::value(k);
            double    p   = b_i[k] / sum_b_i;
            double    p_e = i.value.emit_p(v_k) - p;

            i.value.emit_p.set(v_k, p);

            e += abs(p_e);
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
    virtual double operator () (
        const std::list<std::vector<Y> > &                      obs,
        const std::vector<std::vector<std::vector<double> > > & gamma,
        const std::vector<std::vector<std::vector<double> > > & xi)
    {
        double e = 0.0;

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
    virtual double operator () (
        const std::list<std::vector<Y> > &                      obs,
        const std::vector<std::vector<std::vector<double> > > & gamma,
        const std::vector<std::vector<std::vector<double> > > & xi)
    {
        double e = 0.0;

        assert(!obs.empty());

        m_model.for_each_state([&,this](state_t & i) {
            Y u_i, sigma2_i;

            double f = 0.0;

            // Estimate u
            auto obs_k_iter = obs.begin();
            for (size_t k = 0; obs_k_iter != obs.end(); ++obs_k_iter, ++k) {
                const auto & obs_k = *obs_k_iter;

                for (size_t t = 0; t < obs_k.size(); ++t) {
                    const auto & obs_k_t = obs_k[t];

                    m_model.for_each_trans_from(i,
                    [&,this](const transition_t & ij) {
                        const auto & xi_ij_k = xi[k][ij.value.index];

                        for (size_t l = 0; l < u_i.size(); ++l)
                            u_i[l] += xi_ij_k[t] * obs_k_t[l];

                        f += xi_ij_k[t];
                    });
                }
            }

            for (size_t l = 0; l < u_i.size(); ++l) {
                u_i[l] /= f;

                double p_e = i.value.emit_p.u()[l] - u_i[l];
                e += abs(p_e);
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

                        for (size_t l = 0; l < u_i.size(); ++l) {
                            double delta = u_i[l] - obs_k_t[l];

                            sigma2_i[l] += xi_ij_k[t] * delta * delta;
                        }
                    });
                }
            }

            for (size_t l = 0; l < sigma2_i.size(); ++l) {
                sigma2_i[l] /= f;

                double p_e = i.value.emit_p.sigma2()[l] - sigma2_i[l];
                e += abs(p_e);
            }

            i.value.emit_p.set(u_i, sigma2_i);
        });

        return e;
    }

};  // end of template class baum_welch_emit_gaussian

}}  // end of namespaces impl and math

#endif  // end of #ifndef math__baum_welch_hxx
