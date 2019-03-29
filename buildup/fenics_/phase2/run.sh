#!/bin/bash

script="$(echo $1 | sed s/\.py//)"
count="$2"
log="$3"
success=0
i=0

while [[ "$success" -eq 0 && "$i" -lt "$count" ]]; do
	echo RUNNING ITERATION $i 
	PYTHONPATH=$HOME/mtnlion:$HOME/mtnlion/buildup python3 -c "import $script; print($script.main(time=None, dt=None, plot_time=None, get_test_stats=True))" >> "$log"
	success="$?"
	let i+=1
done

