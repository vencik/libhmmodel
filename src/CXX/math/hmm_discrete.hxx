#ifndef math__hmm_discrete_hxx
#define math__hmm_discrete_hxx

/**
 *  \brief  Random variable with categorial probability distribution
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

#include "math/rng.hxx"
#include "math/categorial_p.hxx"

#include <iostream>
#include <list>
#include <vector>
#include <iostream>
#include <regex>
#include <sstream>
#include <cassert>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

namespace impl {

/** HMM emission with categorial probability distribution */
template <typename Y>
class hmm_emit_categorial {
    private:

    categorial_p<Y> m_impl;  /**< Categorial random variable */

    public:

    /** Probability getter */
    inline double operator () (const Y & y) const { return m_impl(y); }

    /** Random emission getter */
    inline Y rand() const { return m_impl.rand(); }

    /** Probability setter */
    inline void set(const Y & y, double p) { m_impl.set(y, p); }

    /** Random probabilities setter */
    void set_rand() {
        size_t K = Y::cardinality();
        auto   b = rand_p(K);

        for (size_t k = 0; k < K; ++k)
            m_impl.set(Y::value(k), b[k]);
    }

    /**
     *  \brief  Serialiser
     *
     *  \param  out     Output stream
     *  \param  indent  Indentation
     */
    void serialise(std::ostream & out, const std::string & indent = "") const {
        out << indent << "Table size: " << m_impl.size() << std::endl;

        m_impl.for_each([&](const Y & y, double p) {
            out << indent << "\"" << y << "\": " << p << std::endl;
        });
    }

    /**
     *  \brief  Deserialiser
     *
     *  \param  in  Input stream
     *
     *  \return \c true iff deserialisation was successful
     */
    bool deserialise(std::istream & in) {
        m_impl.reset();

        std::smatch bref;

        std::string line;
        size_t      size = 0;

        std::getline(in, line);
        if (std::regex_match(line, bref, std::regex(
            "^[ \\t]*Table size: (\\d+)[ \\t]*$")))
        {
            std::stringstream size_ss(bref[1]);
            size_ss >> size;
        }
        else return false;  // tab size is mandatory

        for (size_t i = 0; i < size; ++i) {
            std::getline(in, line);

            if (std::regex_match(line, bref, std::regex(
                "^[ \\t]*\"(.*)\":[ \\t]*("
                "[-+]?\\d+(\\.\\d+([eE][-+]?\\d+)?)?"
                ")[ \\t]*$")))
            {
                std::stringstream y_ss(bref[1]);
                Y y;
                if ((y_ss >> y).fail()) return false;

                std::stringstream p_ss(bref[2]);
                double p;
                if ((p_ss >> p).fail()) return false;

                set(y, p);
            }
        }

        return true;
    }

};  // end of template class hmm_emit_categorial

}}  // end of namespaces impl and math

#endif  // end of #ifndef math__hmm_discrete_hxx
