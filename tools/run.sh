#!/bin/bash

function help {
        echo "$0 - A tool for running buildup.fenics_ scripts for a given number of iterations."
        echo ""
        echo "$0 [OPTIONS] SCRIPT_NAME NUMBER_ITERATIONS"
        echo ""
        echo "If no arguments are provided, $0 show this help."
        echo ""
        echo "OPTIONS:"
        echo "-h, --help       	show this help"
        echo "-l, --log		save output to a log file, e.g. $0 -l phase1.cs 50 cs.log"
	echo ""
	echo "SCRIPT_NAME	name of the script to run. E.g. phase1.cs or phase2.phie"
	echo "NUMBER_ITERATIONS	number of times to run the script"
}

function run {
	script="$1"
	module=$(echo "$script" | awk '{print $NF}' FS=.)
	count="$2"
	success=0
	i=0

	while [[ "$success" -eq 0 && "$i" -lt "$count" ]]; do
		echo RUNNING ITERATION $i 
		PYTHONPATH="$HOME/mtnlion/:$HOME/mtnlion/buildup" python3 -c "import buildup.fenics_.$script as $module; print($module.main(get_test_stats=True))"
		success="$?"
		let i+=1
	done
}

function log {
	run $1 $2 | tee "$3"
}

if [ $# -gt 0 ]; then
        while [ $# -gt 0 ]; do
        key="$1"
        case $key in
                -h|--help)
                eval
                shift # past argument
                ;;
                -l|--log)
                shift # past argument
		log $1 $2 $3
		exit 0
                ;;
                *)
                run $1 $2
		exit 0
                ;;
        esac
        shift # past argument or value
        done
else
        help
fi
