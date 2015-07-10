/**
 *  Unit test for numerics.
 *
 *  \date    2015/07/05
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

#include "math/numerics.hxx"

#include <exception>
#include <stdexcept>
#include <string>
#include <list>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

#ifndef HAVE_CXX11
#error "Sorry, C++11 support is required to compile this"
#endif


/** Real number in sign * 2^exp representation */
typedef math::sign2exp<double, int> sign2exp_t;


/** math::sign2exp UT */
static int sign2exp_test() {
    std::cout
        << '(' <<  0.0001 << ") - ("
        << sign2exp_t(0.0001) << ')'
        << std::endl;

    std::cout
        << '(' << 0 - -27.768765876 << ") - ("
        << sign2exp_t(0) - -27.768765876 << ')'
        << std::endl;

    std::cout
        << '(' << -27.768765876 + 7654.43645345 << ") - ("
        << sign2exp_t(-27.768765876) + sign2exp_t(7654.43645345) << ')'
        << std::endl;

    std::cout
        << '(' << 0.0002387687 * -7659.76545 << ") - ("
        << sign2exp_t(0.0002387687) * sign2exp_t(-7659.76545) << ')'
        << std::endl;

    std::cout
        << '(' << 0.0002387687 / -7659.76545 << ") - ("
        << sign2exp_t(0.0002387687) / sign2exp_t(-7659.76545) << ')'
        << std::endl;

    std::cout
        << '(' << ::sqrt(654658.000032876) << ") - ("
        << sign2exp_t(654658.000032876).sqrt() << ')'
        << std::endl;

    sign2exp_t::pack_t summands_raw[] = {
        13.456,
        0.123,
        154.09809809,
        0.00000000004356456,
        3.987,
        -100.8765,
        //6547654765476547654765.7665,
        0.00000000000000000000000000000000000876,
        65.00003232,
        0.00000000000023,
        235.87,
    };

    std::list<sign2exp_t> summands;
    for (
        size_t i = 0;
        i < sizeof(summands_raw) / sizeof(summands_raw[0]);
        ++i)
    {
        summands.emplace_back(summands_raw[i]);
    }

    sign2exp_t err;
    sign2exp_t sum_min_err = sign2exp_t::sum(summands, err);
    sign2exp_t sum = sign2exp_t::sum(summands);

    std::cout << '(';

    for (
        size_t i = 0;
        i < sizeof(summands_raw) / sizeof(summands_raw[0]) - 1;
        ++i)
    {
        std::cout << summands_raw[i] << " + ";
    }

    std::cout
        << summands_raw[sizeof(summands_raw) / sizeof(summands_raw[0]) - 1]
        << ") - (" << sum_min_err << " + " << err << ')'
        << std::endl;

    std::cout << '(';

    for (
        size_t i = 0;
        i < sizeof(summands_raw) / sizeof(summands_raw[0]) - 1;
        ++i)
    {
        std::cout << summands_raw[i] << " + ";
    }

    std::cout
        << summands_raw[sizeof(summands_raw) / sizeof(summands_raw[0]) - 1]
        << ") - (" << sum << ')'
        << std::endl;

    return 0;
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        std::cout << /*std::fixed <<*/ std::setprecision(100);

        exit_code = sign2exp_test();
        if (exit_code) break;

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
