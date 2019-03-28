#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
IGNOREDIRS="-type f -path *jitfailure* -prune -o -path ${SCRIPTPATH}/build -prune -o -path ${SCRIPTPATH}/.eggs -prune -o -path ${SCRIPTPATH}/.tox -prune"

function help {
	echo "$0 - A tool for checking and applying format rules"
	echo ""
	echo "$0 [OPTIONS] "
	echo ""
	echo "If no OPTIONS are provided, $0 will run evaluation."
	echo ""
	echo "OPTIONS:"
	echo "-h, --help	show this help"
	echo "-e, --eval	evaluate code for compliance"
}

function eval {
	echo "*** EXECUTING Python LINTER ***"
	declare -a PYFILES=($(find ${SCRIPTPATH} ${IGNOREDIRS} -o \( -iname \*.py \) -print))
	pylint "${PYFILES[@]}"

	echo "*** EXECUTING C++ LINTER ***"
	declare -a CPPFILES=($(find ${SCRIPTPATH} ${IGNOREDIRS} -o \( -name \*.h -o -name \*.cpp \) -print))
	cpplint --quiet "${CPPFILES[@]}"

	echo "*** EXECUTING Shell LINTER ***"
	declare -a SHFILES=($(find ${SCRIPTPATH} ${IGNOREDIRS} -o \( -iname \*.sh \) -print))
	#shellcheck "${SHFILES[@]}"
	echo "Done"
}

if [ $# -gt 0 ]; then
        while [ $# -gt 0 ]; do
        key="$1"
        case $key in
                -e|--eval)
                eval
                shift # past argument
                ;;
                -h|--help)
                help
                exit 0
                ;;
                *)
                help
                exit 0
                ;;
        esac
        shift # past argument or value
        done
else
        eval 
fi

