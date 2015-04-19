# Compiler & linker flags
AM_CXXFLAGS = -g -Wall -Werror
AM_LDFLAGS  = -L../.. -lhmm-gesture

# Unit test scripts
TESTS = \
    hmm.sh \
    baum-welch.sh \
    serialise.sh


# Unit test programs
check_PROGRAMS = \
    baum-welch \
    hmm \
    serialise

baum_welch_SOURCES = \
    baum-welch.cxx

hmm_SOURCES = \
    hmm.cxx

serialise_SOURCES = \
    serialise.cxx
