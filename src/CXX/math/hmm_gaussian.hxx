#ifndef math__hmm_gaussian_hxx
#define math__hmm_gaussian_hxx

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

#include "math/normal_p.hxx"

#include <iostream>
#include <regex>
#include <sstream>
#include <cassert>


#ifndef HAVE_CXX11
#error "Sorry, C++11 support required to compile this"
#endif  // end of #ifndef HAVE_CXX11


namespace math {

namespace impl {

/**
 *  \brief  HMM normal emissions
 */
template <typename Y>
class hmm_emit_gaussian {
    private:

    normal_p<Y> m_impl;  /**< Gaussian random vector */

    /**
     *  \brief  Serialise Y
     *
     *  \param  out     Output stream
     *  \param  indent  Indentation
     */
    static void serialiseY(std::ostream & out, const Y & y) {
        out << '[';

        if (y.size())
            out << y[0];

        for (size_t i = 1; i < y.size(); ++i)
            out << " " << y[i];

        out << ']';
    }

    /**
     *  \brief  Deserialise Y
     *
     *  \param  in  Input stream
     *
     *  \return \c true iff deserialisation was successful
     */
    static bool deserialiseY(std::istream & in, Y & y) {
        static const std::string double_str(
            "([-+]?\\d+(\\.\\d+([eE][-+]?\\d+)?)?)");

        std::string re('[');
        if (y.size())
            re += double_str;
        for (size_t i = 1; i < y.size(); ++i)
            re += " " + double_str;
        re += ']';

        std::smatch bref;
        std::string y_str;
        in >> y_str;

        if (!std::regex_match(y_str, bref, std::regex(re))) return false;

        for (size_t i = 0; i < y.size(); ++i) {
            std::stringstream y_i(bref[i]);

            if ((y_i >> y[i]).fail()) return false;
        }

        return true;
    }

    public:

    /** Probability getter */
    inline double operator () (const Y & y) const { return m_impl(y); }

    /** Random emission getter */
    inline Y rand() const { return m_impl.rand(); }

    /** Emission parameters setter */
    inline void set(const Y & u, const Y & sigma2) {
        m_impl.set(u, sigma2);
    }

    /** Average getter */
    inline const Y & u() const { return m_impl.u(); }

    /** Covariance matrix diagonal getter */
    inline const Y & sigma2() const { return m_impl.sigma2(); }

    /** Random setter */
    inline void set_rand() {
        // Small random u, rather large variance
        Y u, sigma2;

        std::vector<double> init;

        init = math::rand(u.size(), 0.3);
        for (size_t i = 0; i < init.size(); ++i)
            u[i] = init[i];

        init = math::rand(u.size(), 30.0);
        for (size_t i = 0; i < init.size(); ++i)
            sigma2[i] = init[i];

        set(u, sigma2);
    }

    /**
     *  \brief  Serialiser
     *
     *  \param  out     Output stream
     *  \param  indent  Indentation
     */
    void serialise(std::ostream & out, const std::string & indent = "") const {
        out << indent << "Y ~ N" << m_impl.get_u().size() << '(';
        serialiseY(out, m_impl.get_u());
        out << ',';
        serialiseY(out, m_impl.get_sigma2());
        out << ')' << std::endl;
    }

    /**
     *  \brief  Deserialiser
     *
     *  \param  in  Input stream
     *
     *  \return \c true iff deserialisation was successful
     */
    bool deserialise(std::istream & in) {
        std::smatch bref;

        std::string line;
        std::getline(in, line);

        if (!std::regex_match(line, bref, std::regex(
            "^[ \\t]*Y[ \\t]*~[ \\t]*N(\\d+)\\([^,]*),([^\\)]*)\\)[ \\t]*$")))
        {
            return false;
        }

        std::stringstream u_ss(bref[2]);
        std::stringstream sigma2_ss(bref[3]);

        Y u;
        if (!deserialise(u_ss, u)) return false;

        Y sigma2;
        if (!deserialise(sigma2_ss, sigma2)) return false;

        set(u, sigma2);

        return true;
    }

};  // end of template class gaussian_hmm_emission_p

}}  // end of namespace impl and math

#endif  // end of #ifndef math__hmm_gaussian_hxx
