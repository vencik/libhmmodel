#ifndef math__hmm_base_hxx
#define math__hmm_base_hxx

/**
 *  \brief  Hidden Markov Model (base template class)
 *
 *  See http://en.wikipedia.org/wiki/Hidden_Markov_model
 *
 *  IMPLEMENTATION NOTES:
 *  Model states and transitions are indexed (0-based).
 *  This allows for easier and more efficient implementation
 *  of algorithms using the model.
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

#include "container/graph.hxx"

#include "util/io.hxx"

#include <vector>
#include <map>
#include <cassert>
#include <iostream>
#include <regex>
#include <sstream>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

namespace impl {

/**
 *  \brief  Hidden Markov Model (base)
 *
 *  \tparam  X  Hidden random variable
 *  \tparam  Y  Observed random variable
 *  \tparam  P  Emission probability
 */
template <typename X, typename Y, typename P>
class hmm {
    public:

    typedef X hidden_t;    /**< Hidden variable */
    typedef Y emission_t;  /**< Emission        */

    /** Observation */
    typedef std::vector<Y> observation_t;

    /** State value */
    struct state_val {
        size_t    index;  /**< 0-based state index          */
        X             x;  /**< Hidden variable value        */
        double        p;  /**< Start probability            */
        P        emit_p;  /**< Emission probability measure */

        state_val(size_t i, const X & x_init, double p_init):
            index(i),
            x(x_init),
            p(p_init)
        {}

        state_val(size_t i, X && x_init, double p_init):
            index(i),
            x(x_init),
            p(p_init)
        {}

        inline bool operator < (const state_val & rval) const {
            return x < rval.x;
        }

    };  // end of struct state_val

    /** Transition value */
    struct trans_val {
        size_t index;  /**< 0-based transition index */
        double     p;  /**< Transition probability   */

        trans_val(size_t i, double p_init):
            index(i),
            p(p_init)
        {}

        inline bool operator < (const trans_val & rval) const {
            // Note that we want to be able to change the transitions
            // probabilities (during for model learning).
            // Therefore, the operator will simply use index comparison.
            return index < rval.index;
        }

    };  // end of struct trans_val

    /** Model */
    typedef container::dgraph<state_val, trans_val> model_t;

    typedef typename model_t::node   state_t;       /**< Model state */
    typedef typename model_t::branch transition_t;  /**< Transition  */

    private:

    model_t m_impl;  /**< Model structure */

    protected:

    /**
     *  \brief  Add model state
     *
     *  \param  value  State value (aka hidden random variable value)
     *  \param  p      Start probability
     *
     *  \return The state (for transitions creation)
     */
    inline state_t & state(const X & x, double p) {
        return m_impl.add_node(m_impl.node_cnt(), x, p);
    }

    /**
     *  \brief  Add model state
     *
     *  \param  value  State value (aka hidden random variable value)
     *  \param  p      Start probability
     *
     *  \return The state (for transitions creation)
     */
    inline state_t & state(X && x, double p) {
        return m_impl.add_node(m_impl.node_cnt(), x, p);
    }

    /**
     *  \brief  Add transition between model states
     *
     *  \param  o  State of origin
     *  \param  t  Target state
     *  \param  p  Transition probability
     *
     *  \return The transition
     */
    inline transition_t & transition(
        state_t & o,
        state_t & t,
        double    p)
    {
        return m_impl.add_branch(o, t,
            trans_val(m_impl.branch_cnt(), p));
    }

    /**
     *  \brief  Constructor
     *
     *  Descendants MUST provide their own constructor that creates
     *  the model topology and initialises observations probabilities
     *  (via the protected methds above).
     */
    hmm() {}

    public:

    /** Model state count */
    inline size_t state_cnt() const { return m_impl.node_cnt(); }

    /** Model transition count */
    inline size_t transition_cnt() const { return m_impl.branch_cnt(); }

    /** Count of transitions from a state */
    inline size_t transition_cnt_from(const state_t & state) const {
        return m_impl.branch_cnt_from(state);
    }

    /** Count of transitions to a state */
    inline size_t transition_cnt_to(const state_t & state) const {
        return m_impl.branch_cnt_to(state);
    }

    /**
     *  \brief  For-each-state generic algorithm
     *
     *  \tparam I    Injection functor type
     *  \param  inj  Injection functor
     */
    template <class I>
    inline void for_each_state(I inj) {
        m_impl.for_each_node(inj);
    }

    /**
     *  \brief  For-each-state generic algorithm (const)
     *
     *  \tparam I    Injection functor type
     *  \param  inj  Injection functor
     */
    template <class I>
    inline void for_each_state(I inj) const {
        m_impl.for_each_node(inj);
    }

    /**
     *  \brief  For-each-transition generic algorithm
     *
     *  \tparam I    Injection functor type
     *  \param  inj  Injection functor
     */
    template <class I>
    inline void for_each_trans(I inj) {
        m_impl.for_each_branch(inj);
    }

    /**
     *  \brief  For-each-transition generic algorithm (const)
     *
     *  \tparam I    Injection functor type
     *  \param  inj  Injection functor
     */
    template <class I>
    inline void for_each_trans(I inj) const {
        m_impl.for_each_branch(inj);
    }

    /**
     *  \brief  For-each-transition-from-state generic algorithm
     *
     *  \tparam I      Injection functor type
     *  \param  state  State
     *  \param  inj    Injection functor
     */
    template <class I>
    inline void for_each_trans_from(state_t & state, I inj) {
        m_impl.for_each_branch_from(state, inj);
    }

    /**
     *  \brief  For-each-transition-from-state generic algorithm (const)
     *
     *  \tparam I      Injection functor type
     *  \param  state  State
     *  \param  inj    Injection functor
     */
    template <class I>
    inline void for_each_trans_from(const state_t & state, I inj) const {
        m_impl.for_each_branch_from(state, inj);
    }

    /**
     *  \brief  For-each-transition-to-state generic algorithm
     *
     *  \tparam I      Injection functor type
     *  \param  state  State
     *  \param  inj    Injection functor
     */
    template <class I>
    inline void for_each_trans_to(state_t & state, I inj) {
        m_impl.for_each_branch_to(state, inj);
    }

    /**
     *  \brief  For-each-transition-to-state generic algorithm (const)
     *
     *  \tparam I      Injection functor type
     *  \param  state  State
     *  \param  inj    Injection functor
     */
    template <class I>
    inline void for_each_trans_to(const state_t & state, I inj) const {
        m_impl.for_each_branch_to(state, inj);
    }

    /** Reset to empty model */
    inline void reset() { m_impl.reset(); }

    /**
     *  \brief  Serialise model to a stream
     *
     *  \param  out  Output stream
     */
    void serialise(
        const std::string & id,
        std::ostream &      out,
        const std::string & indent = "") const
    {
        out
            << indent << "Model \"" << id << "\"" << std::endl;

        // Serialise states
        for_each_state([&,this](const state_t & state) {
            out
                << indent << "    State " << state.value.index << std::endl
                << indent << "        X = \"" << state.value.x << "\""
                << std::endl
                << indent << "        P = " << state.value.p << std::endl
                << indent << "        Emission" << std::endl;

            state.value.emit_p.serialise(out, indent + "            ");

            out
                << indent << "        EndEmission" << std::endl
                << indent << "    EndState" << std::endl;
        });

        // Serialise transitions
        for_each_trans([&,this](const transition_t & trans) {
            out
                << indent << "    Transition " << trans.value.index
                << std::endl
                << indent << "        from = " << trans.origin().value.index
                << std::endl
                << indent << "        to   = " << trans.target().value.index
                << std::endl
                << indent << "        P    = " << trans.value.p << std::endl
                << indent << "    EndTransition" << std::endl;
        });

        out
            << indent << "EndModel" << std::endl;
    }

    private:

    /**
     *  \brief  Deserialise state (definition inner level)
     *
     *  \param  in      Input stream
     *  \param  index   State index
     *  \param  states  Vector of states
     *
     *  \return \c true iff deserialisation was successful
     */
    bool deserialise_state(
        std::istream &           in,
        size_t                   index,
        std::vector<state_t *> & states)
    {
        state_t & state = m_impl.add_node(index, X(), 0.0);

        for (size_t size = states.size(); !(index < size); ++size)
            states.push_back(NULL);

        if (NULL != states[index])
            return false;  // state index duplicity

        states[index] = &state;

        std::smatch bref;  // back-references

        for (;;) {
            std::string line(util::readline(in));

            // Done
            if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*EndState[ \\t]*$")))
            {
                return true;
            }

            // Hidden random variable
            else if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*X[ \\t]*=[ \\t]*\"(.*)\"[ \\t]*$")))
            {
                std::stringstream x_ss(bref[1]);

                x_ss >> state.value.x;
            }

            // Initial probability
            else if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*P[ \\t]*=[ \\t]*"
                "([-+]?\\d+(\\.\\d+([eE][-+]?\\d+)?)?)"
                "[ \\t]*$")))
            {
                std::stringstream p_ss(bref[1]);

                p_ss >> state.value.p;
            }

            // Emission
            else if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*Emission[ \\t]*$")))
            {
                if (!state.value.emit_p.deserialise(in))
                    return false;

                line = util::readline(in);

                if (!std::regex_match(line, bref, std::regex(
                    "^[ \\t]*EndEmission[ \\t]*$")))
                {
                    return false;
                }
            }

            // Syntax error
            else return false;
        }
    }

    /**
     *  \brief  Retrieve state by index (still in string form)
     *
     *  \param  states     Vector of states
     *  \param  index_str  State index (as string)
     *
     *  \return State pointer or \c NULL in case of error
     */
    static state_t * state(
        const std::vector<state_t *> & states,
        const std::string &            index_str)
    {
        std::stringstream index_ss(index_str);
        size_t index;

        if ((index_ss >> index).fail()) return NULL;

        if (!(index < states.size())) return NULL;

        return states[index];
    }

    /**
     *  \brief  Deserialise transition (definition inner level)
     *
     *  \param  in      Input stream
     *  \param  index   Transition index
     *  \param  states  Vector of states
     *
     *  \return \c true iff deserialisation was successful
     */
    bool deserialise_trans(
        std::istream &           in,
        size_t                   index,
        std::vector<state_t *> & states)
    {
        std::smatch bref;  // back-references

        state_t * origin = NULL;
        state_t * target = NULL;
        double    p      = 0.0;

        for (;;) {
            std::string line(util::readline(in));

            // Done
            if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*EndTransition[ \\t]*$")))
            {
                if (NULL == origin || NULL == target) return false;

                m_impl.add_branch(*origin, *target,
                    trans_val(index, p));

                return true;
            }

            // Origin
            else if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*from[ \\t]*=[ \\t]*(.*)[ \\t]*$")))
            {
                origin = state(states, bref[1]);
            }

            // Target
            else if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*to[ \\t]*=[ \\t]*(.*)[ \\t]*$")))
            {
                target = state(states, bref[1]);
            }

            // Trans. probability
            else if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*P[ \\t]*=[ \\t]*"
                "([-+]?\\d+(\\.\\d+([eE][-+]?\\d+)?)?)"
                "[ \\t]*$")))
            {
                std::stringstream p_ss(bref[1]);

                p_ss >> p;
            }

            // Syntax error
            else return false;
        }
    }

    /**
     *  \brief  Deserialise model (definition inner level)
     *
     *  \param  in  Input stream
     *
     *  \return \c true iff deserialisation was successful
     */
    bool deserialise_model(std::istream & in) {
        // States vector
        std::vector<state_t *> states;

        std::smatch bref;  // back-references

        for (;;) {
            std::string line(util::readline(in));

            // Done
            if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*EndModel[ \\t]*$")))
            {
                return true;
            }

            // State definition
            else if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*State[ \\t]+(\\d+)[ \\t]*$")))
            {
                deserialise_state(in, (size_t)std::stoul(bref[1]), states);
            }

            // Transition definition
            else if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*Transition[ \\t]+(\\d+)[ \\t]*$")))
            {
                deserialise_trans(in, (size_t)std::stoul(bref[1]), states);
            }

            // Syntax error
            else return false;
        }
    }

    public:

    /**
     *  \brief  Deserialise model from input stream
     *
     *  \param  in  Input stream
     *
     *  \return \c true iff deserialisation was successful
     */
    bool deserialise(std::istream & in) {
        reset();

        std::smatch bref;  // back-references

        // Model <id>
        std::string line(util::readline(in));

        if (std::regex_match(line, bref, std::regex(
            "^[ \\t]*Model[ \\t]+\"([^\"]*)\"[ \\t]*$")))
        {
            return deserialise_model(in);
        }

        return false;
    }

};  // end of template class hmm

}}  // end of namespaces impl and math

#endif  // end of #ifndef math__hmm_base_hxx
