# Compiler & linker flags
AM_CXXFLAGS = -g -Wall -Werror
AM_LDFLAGS  =

# Unit test scripts
TESTS = \
    numerics.sh \
    normal_p.sh \
    hmm.sh \
    baum-welch.sh \
    serialise.sh \
    ghmm.sh \
    baum-welch.ghmm.sh


# Unit test programs
check_PROGRAMS = \
    numerics \
    normal_p \
    baum-welch \
    hmm \
    serialise \
    ghmm \
    baum-welch.ghmm

numerics_SOURCES = \
    numerics.cxx

normal_p_SOURCES = \
    normal_p.cxx

baum_welch_SOURCES = \
    baum-welch.cxx

hmm_SOURCES = \
    hmm.cxx

serialise_SOURCES = \
    serialise.cxx

ghmm_SOURCES = \
    ghmm.cxx

baum_welch_ghmm_SOURCES = \
    baum-welch.ghmm.cxx
