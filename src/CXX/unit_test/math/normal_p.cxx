/**
 *  Unit test for multivariate normal distribution
 *
 *  \date    2015/06/30
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
#include "math/numerics.hxx"

#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cassert>

#ifndef HAVE_CXX11
#error "Sorry, C++11 support is required to compile this"
#endif


/** N-dimensional variable */
template <size_t N>
class X: public math::real_vector {
    public:

    /** Rank */
    static size_t rank() { return N; }

    /** Default constructor */
    X(): math::real_vector(N) {}

    /**
     *  \brief  Constructor
     *
     *  \param  i  Initialiser
     */
    X(const math::real_t & i): math::real_vector(N, i) {}

    /**
     *  \brief  Constructor
     *
     *  \param  il  Initialiser list
     */
    X(const std::initializer_list<math::real_t> & il):
        math::real_vector(il) {}

};  // end of class X

/** 2D variable */
typedef X<2> X_t;


/** 2D random variable with normal distr. of probability */
typedef math::normal_p<X_t> normal_p_t;


/** Probability of rand. variable */
static void p(const normal_p_t & n, const X_t & x) {
    std::cout
        << "P(X ~ N(" << n.u() << ", " << n.sigma2()
        << ") =~ " << x << ") == " << n(x)
        << std::endl;
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        normal_p_t n({1.0, 0.0}, {0.5, 0.3});

        // Get probability of close value
        p(n, { 1.1,  0.1});
        p(n, { 1.6,  0.0});
        p(n, { 2.0,  0.0});
        p(n, { 0.0,  1.0});
        p(n, {-1.0, -1.0});
        p(n, { 8.0,  4.0});

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
