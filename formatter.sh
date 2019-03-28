#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
IGNOREDIRS="-type f -path *jitfailure* -prune -o -path .tox -prune"

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
	echo "-f, --format	perform format on all files in path"
}

function eval {
	echo "*** EXECUTING Python Evaluator ***"
	declare -a PYFILES=($(find ${SCRIPTPATH} ${IGNOREDIRS} -o \( -iname \*.py \) -print))
	black --check --py36 "${PYFILES[@]}"

	echo "*** EXECUTING C++ Evaluator ***"
	declare -a CPPFILES=($(find ${SCRIPTPATH} ${IGNOREDIRS} -o \( -name \*.h -o -name \*.cpp \) -print))
	for CPPFILE in "${CPPFILES[@]}"
	do
	  OUTPUT=$(clang-format -style=Google -output-replacements-xml "$CPPFILE" | grep -c "<replacement " >/dev/null || echo Ok)
	  if [ "$OUTPUT" != "Ok" ]
	  then
	    echo "$CPPFILE"
	  fi
	done

	echo Done!
}

function format {
	echo "*** EXECUTING Python Evaluator ***"
	declare -a PYFILES=($(find ${SCRIPTPATH} -path .tox -prune -o \( -iname \*.py \) -print))
	black --py36 "${PYFILES[@]}"

	echo "*** EXECUTING C++ Evaluator ***"
	declare -a CPPFILES=($(find ${SCRIPTPATH} -path .tox -prune -o \( -name \*.h -o -name \*.cpp \) -print))
	for CPPFILE in "${CPPFILES[@]}"
	do
	  clang-format -style=Google -i "$CPPFILE" 
	done

	echo Done!
}

if [ $# -gt 0 ]; then
        while [ $# -gt 0 ]; do
        key="$1"
        case $key in
                -e|--eval)
                eval
                shift # past argument
                ;;
                -f|--format)
                format
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
