#!/bin/bash

echo "*** EXECUTING Python LINTER ***"
declare -a PYFILES=($(find . -path .tox -prune -o \( -iname \*.py \) -print))
pylint "${PYFILES[@]}"

echo "*** EXECUTING C++ LINTER ***"
declare -a CPPFILES=($(find . -path .tox -prune -o \( -name \*.h -o -name \*.cpp \) -print))
cpplint --quiet "${CPPFILES[@]}"

echo "*** EXECUTING Shell LINTER ***"
declare -a SHFILES=($(find . -path .tox -prune -o \( -iname \*.sh \) -print))
shellcheck "${SHFILES[@]}"

echo "Done"

