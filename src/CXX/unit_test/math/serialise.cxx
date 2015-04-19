/**
 *  Hmm (de)serialisation unit test
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
#include <iostream>
#include <cassert>
#include <cstring>


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
std::ostream & operator << (
    std::ostream & out, state_t state)
{
    return out << state_t_str[state];
}

/** State deserialisation */
static std::istream & operator >> (
    std::istream & in, state_t & state)
{
    char buff[8];

    in.get(buff, 8, '\0');  // read 7 characters

    char i = buff[6];

    assert(0 == ::strncmp("State ", buff, 6));
    assert('1' == i || '2' == i);

    state = (state_t)(i - '1');

    return in;
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

/** Emission deserialisation */
std::istream & operator >> (
    std::istream & in, emission_value_t & e)
{
    char buff[8];

    in.get(buff, 5, '\0');  // get 4 characters

    // No eggs
    if ('N' == buff[0]) {
        in.get(buff + 4, 4, '\0');  // get remaining 3 characters

        assert(0 == ::strcmp("No eggs", buff));

        e = NO_EGGS;
    }

    else {
        assert(0 == ::strcmp("Eggs", buff));

        e = EGGS;
    }

    return in;
}

/** Output random variable (wrapper) */
class emission_t: public math::hmm_emission_t<emission_value_t> {
    public:

    /** Default constructor */
    emission_t():
        math::hmm_emission_t<emission_value_t>(NO_EGGS)
    {}

    /** Constructor */
    emission_t(emission_value_t v):
        math::hmm_emission_t<emission_value_t>(v)
    {}

    static size_t cardinality() { return 2; }

    static emission_t value(size_t index) {
        return emission_t((emission_value_t)index);
    }

};  // end of class emission_t


/** Model */
typedef math::hmm<state_t, emission_t> model_t;


/** Model (de)serialisation unit test */
static int serialise_test(const std::string & model_id) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        model_t m(std::cin);

        m.serialise(model_id, std::cout);

        exit_code = 0;  // success

    } while (0);  // end of pragmatic loop

    return exit_code;
}

/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    std::string model_id;

    if (argc > 1) model_id = argv[1];

    int exit_code = serialise_test(model_id);

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
            << std::endl;
    }
    catch (...) {
        std::cerr
            << "Unhandled non-standard exception caught"
            << std::endl;
    }

    return exit_code;
}
