# Hidden Markov Model library

AC_PREREQ([2.68])
AC_INIT([libhmmodel], [0.1.0], [vencik@razdva.cz])
AM_INIT_AUTOMAKE
AM_SILENT_RULES([yes])
LT_INIT
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/CXX/math/hmm.hxx])  # TODO: use project main .cxx when created
AC_CONFIG_HEADERS([src/CXX/config.hxx])
AC_PREFIX_DEFAULT([/usr/local])


#
# Resolve arguments
#

# Set optimisation level
AC_MSG_CHECKING([for optimisation level])
AC_ARG_WITH([optim-level],
    AS_HELP_STRING([--with-optim-level], [Set compiler optimisations level (default: 2)]),
    [   # --with-optim-level specified
        case "${withval}" in
            ""|0|1|2|3|s|fast)
                AC_MSG_RESULT([$withval])
                CFLAGS="${CXXFLAGS} -O${withval}"
                ;;
            *)
                AC_MSG_ERROR([unexpected --with-optim-level argument: ${withval}])
                CFLAGS="${CXXFLAGS} -O2"
                ;;
        esac
    ],
    [   # --with-optim-level not specified
        AC_MSG_RESULT([2])
        CFLAGS="${CXXFLAGS} -O2"
    ])

# Strip binaries
AC_MSG_CHECKING([whether to strip binaries])
AC_ARG_ENABLE([strip],
    AS_HELP_STRING([--enable-strip], [Strip debug info from binaries (default: no)]),
    [   # --enable-strip specified (with or without argument)
        case "${enableval}" in
            no|false|off)
                AC_MSG_RESULT([no])
                ;;
            yes|true|on|"")
                AC_MSG_RESULT([yes])
                CFLAGS="${CXXFLAGS} -s"
                ;;
            *)
                AC_MSG_ERROR([unexpected --enable-strip argument: ${enableval}])
                ;;
        esac
    ],
    [   # --enable-strip not specified
        AC_MSG_RESULT([no])
    ])


#
# Checks for programs
#

AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_MAKE_SET

# AsciiDoc is required for documentation
AM_COND_IF([ENABLE_DOC], [
    AC_PATH_PROG([ASCIIDOC], [asciidoc], [])
    test -n "${ASCIIDOC}" || AC_MSG_ERROR([AsciiDoc is required for doc build (--disable-doc will help)])

    AC_PATH_PROG([A2X], [a2x], [])
    if test -n "${A2X}"; then
        enable_doc_pub=true
    else
        AC_MSG_WARN([a2x not found; printable docs will not be built])
    fi
])
AM_CONDITIONAL([ENABLE_DOC_PUB], [test x$enable_doc_pub = xtrue])


#
# Checks for libraries
#


#
# Checks for typedefs, structures, and compiler characteristics
#

AC_HEADER_STDBOOL
AC_C_INLINE

# We use C++11
AX_CXX_COMPILE_STDCXX_11([noext])


#
# Checks for library functions
#


#
# Create files
#

AC_CONFIG_FILES([
    Makefile
    src/Makefile
    src/CXX/Makefile
    src/CXX/container/Makefile
    src/CXX/math/Makefile
    src/CXX/unit_test/Makefile
    src/CXX/unit_test/container/Makefile
    src/CXX/unit_test/math/Makefile
])
AC_OUTPUT
