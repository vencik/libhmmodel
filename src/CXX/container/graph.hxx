#ifndef container__graph_hxx
#define container__graph_hxx

/**
 *  \brief  Graph container
 *
 *  Graph container for accomodation of the HMM topology.
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

#include <set>
#include <cassert>
#include <cstdlib>


namespace container {

/**
 *  \brief  Directed graph
 *
 *  Dynamic graph structure allowing for evaluation
 *  of both nodes and branches.
 *
 *  This implementation allows for branch multiplicity
 *  (i.e. more than 1 branch from node A to node B).
 *
 *  IMPLEMENTATION NOTES
 *
 *  Nodes are held in a multi-set ordered by the node value.
 *  Branches are held in a multi-set ordered by the branch value.
 *  Each node also has 2 multi-sets of references to branches that
 *  originate in it and that target it.
 *  This allows for fast iteration over branches from/to a node
 *  as well as efficient selection of optimal branches.
 */
template <typename N, typename B>
class dgraph {
    public:

    // Forward declarations */
    /** \cond */
    struct node;
    struct branch;
    /** \endcond */

    private:

    /** Set of nodes */
    typedef std::multiset<node> nodes_t;

    /** Set of branches (organised by value) */
    typedef std::multiset<branch> branches_t;

    /** Reference to node */
    typedef typename nodes_t::iterator node_ref_t;

    /** Reference to const node */
    typedef typename nodes_t::const_iterator const_node_ref_t;

    /** Reference to branch */
    typedef typename branches_t::iterator branch_ref_t;

    /** Reference to const branch */
    typedef typename branches_t::const_iterator const_branch_ref_t;

    /** Wrapper around branch reference (to enable sets) */
    struct branch_ref_w {
        friend class dgraph;

        branch_ref_t impl;

        branch_ref_w(branch_ref_t ref): impl(ref) {}

        inline bool operator < (const branch_ref_w & br) const {
            return *impl < *br.impl;
        }

        operator const branch_ref_t & () const { return impl; }

    };  // end of struct branch_ref_w

    /** Set of branch references */
    typedef std::multiset<branch_ref_w> branch_refs_t;

    /** Reference to branch ref. */
    typedef typename branch_refs_t::iterator branch_ref2_t;

    /** Reference to const branch ref. */
    typedef typename branch_refs_t::const_iterator const_branch_ref2_t;

    public:

    /** Graph node */
    class node {
        friend class dgraph;

        private:

        branch_refs_t m_branches_from;  /**< Branches from this node */
        branch_refs_t m_branches_to;    /**< Branches to   this node */

        node_ref_t m_ref;  /**< Reference in node set */

        public:

        N value;  /**< Node value */

        node(const N & val): value(val) {}

#ifdef HAVE_CXX11
        node(N && rval): value(std::move(rval)) {}
#endif

        /** Comparison (for std::set) */
        inline bool operator < (const node & n) const {
            return value < n.value;
        }

    };  // end of class node

    /** Graph branch */
    class branch {
        friend class dgraph;

        private:

        node & m_origin;  /**< Node of origin */
        node & m_target;  /**< Target node    */

        branch_ref_t  m_ref;       /**< Branch set ref.      */
        branch_ref2_t m_ref_from;  /**< Origin branches ref. */
        branch_ref2_t m_ref_to;    /**< Target branches ref. */

        /** References setter */
        inline void set_refs(
            branch_ref_t  ref,
            branch_ref2_t from,
            branch_ref2_t to)
        {
            m_ref      = ref;
            m_ref_from = from;
            m_ref_to   = to;
        }

        public:

        B value;  /**< Branch value */

        branch(
            node &    origin,
            node &    target,
            const B & val)
        :
            m_origin(origin),
            m_target(target),
            value(val)
        {}

#ifdef HAVE_CXX11
        branch(
            node & origin,
            node & target,
            B &&   rval)
        :
            m_origin(origin),
            m_target(target),
            value(std::move(rval))
        {}

        branch(branch && rval):
            m_origin(rval.m_origin),
            m_target(rval.m_target),
            value(std::move(rval.value))
        {}
#endif

        /** Comparison (for std::multiset) */
        inline bool operator < (const branch & b) const {
            return value < b.value;
        }

        /** Node of origin */
        inline node & origin() { return m_origin; }

        /** Node of origin (const) */
        inline const node & origin() const { return m_origin; }

        /** Targert node */
        inline node & target() { return m_target; }

        /** Targert node (c0nst) */
        inline const node & target() const { return m_target; }

    };  // end of class branch

    private:

    nodes_t    m_nodes;     /**< Set of nodes    */
    branches_t m_branches;  /**< Set of branches */

    /**
     *  \brief  Add node
     *
     *  \param  ref  Node reference
     *
     *  \return Node
     */
    inline node & add_node_impl(node_ref_t ref) {
        node & n = const_cast<node &>(*ref);
        n.m_ref  = ref;

        return n;
    }

    public:

    /**
     *  \brief  Add node
     *
     *  \param  value  Node value
     *
     *  \return Node
     */
    inline node & add_node(const N & value) {
        return add_node_impl(
#ifdef HAVE_CXX11
            m_nodes.emplace(value)
#else
            m_nodes.insert(node(value))
#endif  // end of #ifdef HAVE_CXX11
        );
    }


#ifdef HAVE_CXX11
    /**
     *  \brief  Add node (move value)
     *
     *  \param  rval  Node value (temporary)
     *
     *  \return Nodesize(); }
     */
    inline node & add_node(N && value) {
        return add_node_impl(
            m_nodes.emplace(std::move(value)));
    }

    /**
     *  \brief  Emplace node
     *
     *  \param  args  Node value constructor arguments
     *
     *  \return Node
     */
    template <typename... Cargs>
    inline node & add_node(Cargs && ... args) {
        return add_node_impl(
            m_nodes.emplace(N(args...)));
    }
#endif  // end of #ifdef HAVE_CXX11

    private:

    /**
     *  \brief  Add branch (implementation of refs setting)
     *
     *  \param  bref  Branch reference
     *
     *  \return Branch
     */
    inline branch & add_branch_impl(branch_ref_t bref) {
        branch_ref2_t ref_from =
#ifdef HAVE_CXX11
            bref->m_origin.m_branches_from.emplace(bref);
#else
            bref->m_origin.m_branches_from.insert(branch_ref_w(bref));
#endif  // end of #ifdef HAVE_CXX11

        branch_ref2_t ref_to =
#ifdef HAVE_CXX11
            bref->m_target.m_branches_to.emplace(bref);
#else
            bref->m_target.m_branches_to.insert(branch_ref_w(bref));
#endif  // end of #ifdef HAVE_CXX11

        branch & b = const_cast<branch &>(*bref);

        b.set_refs(bref, ref_from, ref_to);

        return b;
    }

    public:

    /**
     *  \brief  Add branch
     *
     *  \param  origin  Node of origin
     *  \param  target  Target node
     *  \param  value   Branch value
     *
     *  \return Branch
     */
    inline branch & add_branch(
        node &    origin,
        node &    target,
        const B & value)
    {
        return add_branch_impl(
#ifdef HAVE_CXX11
            m_branches.emplace(origin, target, value)
#else
            m_branches.insert(branch(origin, target, value))
#endif  // end of #ifdef HAVE_CXX11
        );
    }

#ifdef HAVE_CXX11
    /**
     *  \brief  Add branch (move value)
     *
     *  \param  origin  Node of origin
     *  \param  target  Target node
     *  \param  rval    Branch value (temporary)
     *
     *  \return Branch
     */
    inline branch & add_branch(
        node & origin,
        node & target,
        B &&   value)
    {
        return add_branch_impl(
            m_branches.emplace(origin, target, std::move(value)));
    }

    /**
     *  \brief  Emplace branch
     *
     *  \param  origin  Node of origin
     *  \param  target  Target node
     *  \param  args    Branch value constructor arguments
     *
     *  \return Branch
     */
    template <typename... Cargs>
    inline branch & add_branch(
        node &       origin,
        node &       target,
        Cargs && ... args)
    {
        return add_branch_impl(
            m_branches.emplace(origin, target, B(args...)));
    }
#endif  // end of #ifdef HAVE_CXX11

    /**
     *  \brief  Remove branch
     *
     *  \param  b  Branch
     */
    inline void remove_branch(branch & b) {
        b.m_origin.m_branches_from.erase(b.m_ref_from);
        b.m_target.m_branches_to.erase(b.m_ref_to);
        m_branches.erase(b.m_ref);
    }

    /**
     *  \brief  Remove node (and its branches)
     *
     *  Note that any handles of branches originating
     *  in this node become invalid.
     *
     *  \param  n  Node
     */
    inline void remove_node(node & n) {
        branch_ref2_t bref2;
        branch_ref2_t bend2;

        // Remove branches from the node
        bref2 = n.m_branches_from.begin();
        bend2 = n.m_branches_from.end();

        for (; bref2 != bend2; ++bref2) {
            branch_ref_t bref = *bref2;

            assert(&bref->m_origin == &n);

            bref->m_target.m_branches_to.erase(bref->m_ref_to);

            m_branches.erase(bref);
        }

        // Remove branches to the node
        bref2 = n.m_branches_to.begin();
        bend2 = n.m_branches_to.end();

        for (; bref2 != bend2; ++bref2) {
            branch_ref_t bref = *bref2;

            assert(&bref->m_target == &n);

            bref->m_origin.m_branches_from.erase(bref->m_ref_from);

            m_branches.erase(bref);
        }

        m_nodes.erase(n.m_ref);
    }

    /**
     *  \brief  For-each-node generic algorithm
     *
     *  Perform injection for all graph nodes.
     *  The injection functor takes \ref node& as argument
     *  and returns an integer.
     *
     *  \param  inj  Injection (action performed per node)
     */
    template <class I>
    inline void for_each_node(I inj) {
        node_ref_t nref = m_nodes.begin();
        node_ref_t nend = m_nodes.end();

        for (; nref != nend; ++nref)
            inj(const_cast<node &>(*nref));
    }

    /**
     *  \brief  For-each-node generic algorithm (const)
     *
     *  Perform injection for all graph nodes.
     *  The injection functor takes \ref node& as argument
     *  and returns an integer.
     *
     *  \param  inj  Injection (action performed per node)
     */
    template <class I>
    inline void for_each_node(I inj) const {
        const_node_ref_t nref = m_nodes.begin();
        const_node_ref_t nend = m_nodes.end();

        for (; nref != nend; ++nref)
            inj(*nref);
    }

    /**
     *  \brief  For-each-branch generic algorithm
     *
     *  Perform injection for all graph branches.
     *  The injection functor takes \ref branch& as argument
     *  and returns an integer.
     *
     *  \param  inj  Injection (action performed per branch)
     */
    template <class I>
    inline void for_each_branch(I inj) {
        branch_ref_t bref = m_branches.begin();
        branch_ref_t bend = m_branches.end();

        for (; bref != bend; ++bref)
            inj(const_cast<branch &>(*bref));
    }

    /**
     *  \brief  For-each-branch generic algorithm (const)
     *
     *  Perform injection for all graph branches.
     *  The injection functor takes \ref branch& as argument
     *  and returns an integer.
     *
     *  \param  inj  Injection (action performed per branch)
     */
    template <class I>
    inline void for_each_branch(I inj) const {
        const_branch_ref_t bref = m_branches.begin();
        const_branch_ref_t bend = m_branches.end();

        for (; bref != bend; ++bref)
            inj(*bref);
    }

    /**
     *  \brief  For-each-branch-from-node generic algorithm
     *
     *  Perform injection for all branches originating in \c nod node.
     *  The injection functor takes \ref branch& as argument
     *  and returns an integer.
     *
     *  \param  nod  Graph node
     *  \param  inj  Injection (action performed per branch)
     */
    template <class I>
    inline void for_each_branch_from(node & nod, I inj) {
        branch_ref2_t bref2 = nod.m_branches_from.begin();
        branch_ref2_t bend2 = nod.m_branches_from.end();

        for (; bref2 != bend2; ++bref2) {
            branch_ref_t bref = *bref2;

            inj(const_cast<branch &>(*bref));
        }
    }

    /**
     *  \brief  For-each-branch-from-node generic algorithm (const)
     *
     *  Perform injection for all branches originating in \c nod node.
     *  The injection functor takes \ref branch& as argument
     *  and returns an integer.
     *
     *  \param  nod  Graph node
     *  \param  inj  Injection (action performed per branch)
     */
    template <class I>
    inline void for_each_branch_from(const node & nod, I inj) const {
        const_branch_ref2_t bref2 = nod.m_branches_from.begin();
        const_branch_ref2_t bend2 = nod.m_branches_from.end();

        for (; bref2 != bend2; ++bref2) {
            const_branch_ref_t bref = *bref2;

            inj(*bref);
        }
    }

    /**
     *  \brief  For-each-branch-to-node generic algorithm
     *
     *  Perform injection for all branches targetting \c nod node.
     *  The injection functor takes \ref branch& as argument
     *  and returns an integer.
     *
     *  \param  nod  Graph node
     *  \param  inj  Injection (action performed per branch)
     */
    template <class I>
    inline void for_each_branch_to(node & nod, I inj) {
        branch_ref2_t bref2 = nod.m_branches_to.begin();
        branch_ref2_t bend2 = nod.m_branches_to.end();

        for (; bref2 != bend2; ++bref2) {
            branch_ref_t bref = *bref2;

            inj(const_cast<branch &>(*bref));
        }
    }

    /**
     *  \brief  For-each-branch-to-node generic algorithm (const)
     *
     *  Perform injection for all branches targetting \c nod node.
     *  The injection functor takes \ref branch& as argument
     *  and returns an integer.
     *
     *  \param  nod  Graph node
     *  \param  inj  Injection (action performed per branch)
     */
    template <class I>
    inline void for_each_branch_to(const node & nod, I inj) const {
        const_branch_ref2_t bref2 = nod.m_branches_to.begin();
        const_branch_ref2_t bend2 = nod.m_branches_to.end();

        for (; bref2 != bend2; ++bref2) {
            const_branch_ref_t bref = *bref2;

            inj(*bref);
        }
    }

    /** Node count */
    inline size_t node_cnt() const { return m_nodes.size(); }

    /** Branch count */
    inline size_t branch_cnt() const { return m_branches.size(); }

    /** Branch count (from a node) */
    inline size_t branch_cnt_from(const node & n) const {
        return n.m_branches_from.size();
    }

    /** Branch count (to a node) */
    inline size_t branch_cnt_to(const node & n) const {
        return n.m_branches_to.size();
    }

    /** Branches from a node exist */
    inline bool branches_from(const node & n) const {
        return !n.m_branches_from.empty();
    }

    /** Branches to a node exist */
    inline bool branches_to(const node & n) const {
        return !n.m_branches_to.empty();
    }

    /** Get 1st (by branch order, i.e. optimal) branch from a node */
    inline branch & opt_branch_from(node & n) {
        return *n.m_branches_from.begin();
    }

    /** Get 1st (by branch order, i.e. optimal) branch from a node (const) */
    inline const branch & opt_branch_from(const node & n) const {
        return *n.m_branches_from.begin();
    }

    /** Get 1st (by branch order, i.e. optimal) branch to a node */
    inline branch & opt_branch_to(node & n) {
        return *n.m_branches_to.begin();
    }

    /** Get 1st (by branch order, i.e. optimal) branch to a node (const) */
    inline const branch & opt_branch_to(const node & n) const {
        return *n.m_branches_to.begin();
    }

    /** Reset graph */
    inline void reset() {
        m_branches.clear();
        m_nodes.clear();
    }

};  // end of template class dgraph

}  // end of namespace container

#endif  // end of #ifndef container__graph_hxx
