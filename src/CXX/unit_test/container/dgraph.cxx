/**
 *  \brief  Graph container unit test
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

#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>
#include <typeinfo>
#include <cassert>

#ifndef HAVE_CXX11
#error "Sorry, C++11 support is required to compile this"
#endif


/** Type wrapper (for tracking purposes) */
template <typename T>
class wrapper {
    private:

    static const std::string s_type_name;  /**< Wrapped type name          */
    static unsigned          s_next_id;    /**< Wrapper object ID provider */

    unsigned m_id;    /**< Object ID     */
    const T  m_impl;  /**< Wrapped value */

    public:

    static bool logging_on;  /**< Log wrapper operations */

    void log(const std::string & msg) {
        if (!logging_on) return;

        std::cerr
            << "wrapper<" << s_type_name
            << ">(ID=0x" << std::hex << m_id << std::dec << ", "
            << "value=" << m_impl
            << "): " << msg
            << std::endl;
    }

    explicit wrapper(const T & value):
        m_id(s_next_id++),
        m_impl(value)
    {
        log("created");
    }

    wrapper(const wrapper & orig):
        m_id(s_next_id++),
        m_impl(orig.m_impl)
    {
        log("copied");

#ifdef HAVE_CXX11
        // We want no copying
        throw std::logic_error(
            "Unnecessary use of copy constructor");
#endif  // end of #ifdef HAVE_CXX11
    }

#ifdef HAVE_CXX11
    wrapper(wrapper && rval):
        m_id(rval.m_id),
        m_impl(rval.m_impl)
    {
        log("moved");

        rval.m_id = 0;  // mark moved tmp. object
    }
#endif  // end of #ifdef HAVE_CXX11

    inline operator const T & () const { return m_impl; }

    inline std::ostream & operator << (std::ostream & out) const {
        return out << m_impl;
    }

    ~wrapper() {
        // Destruction of moved tmp. objects is not interesting
        if (0 == m_id) return;

        log("destroyed");
    }

};  // end of class wrapper

// wrapper<T> static members initialisation
template <typename T>
const std::string wrapper<T>::s_type_name(typeid(T).name());
template <typename T>
unsigned wrapper<T>::s_next_id = 0x1;
template <typename T>
bool wrapper<T>::logging_on = false;


/** Graph type */
typedef container::dgraph<wrapper<int>, wrapper<float> > graph_t;

/** Dump graph to output stream */
static void dump_graph(graph_t & graph, std::ostream & out) {
    graph.for_each_node([&](graph_t::node & n) {
        out
            << "Node " << n.value
            << std::endl;

        graph.for_each_branch_from(n, [&](graph_t::branch & b) {
            out
                << "    "
                << "Branch " << b.value
                << " to node " << b.target().value
                << std::endl;
        });

        graph.for_each_branch_to(n, [&](graph_t::branch & b) {
            out
                << "    "
                << "Branch " << b.value
                << " from node " << b.origin().value
                << std::endl;
        });
    });
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        // Create graph
        graph_t graph;

        graph_t::node & n1 = graph.add_node(1);
        graph_t::node & n2 = graph.add_node(2);
        graph_t::node & n3 = graph.add_node(3);

        graph.add_branch(n1, n2, 1.2);
        graph.add_branch(n1, n3, 1.3);
        graph.add_branch(n2, n3, 2.3);
        graph.add_branch(n3, n2, 3.2);
        graph.add_branch(n1, n1, 1.1);

        dump_graph(graph, std::cout);

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
