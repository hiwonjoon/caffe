#! /bin/bash
if [ $# -ne 1 ]
then
	echo "Specify the directory which you want."
	exit 1
fi
cd $1

count=($(ls -1 ./ | grep .solverstate | wc -l))
filename=$(date +"%F_%H_%M")
echo $filename

if [ $count -ge "1" ]
then
	list=($(ls -1 ./*.solverstate | tr '\n' '\0' | xargs -0 -n 1 basename | sort -V -r))
	read -p "You have a solverstate. Do you want to continue learning process from the last(y/n)? " answer
	case ${answer:0:1} in
		y|Y )
			../../build/tools/caffe train -solver ./solver.prototxt -gpu 0 -snapshot ./$list &> $filename.log &
		;;
		* )
			../../build/tools/caffe train -solver ./solver.prototxt -weights ../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0 &> $filename.log &
		;;
	esac
else
	../../build/tools/caffe train -solver ./solver.prototxt -weights ../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0 &> $filename.log &
fi

cd ..

#script for future use

#!/bin/bash
#list=$(ls -1 ./regularized_fix/*.solverstate | tr '\n' '\0' | xargs -0 -n 1 basename)
#for file in $list
#do
#	echo $file
#done
#files=./regularized_fix/"*.solverstate"
#regex='([0-9]+)\.solverstate'
#for f in $files
#do
#	[[ $f =~ $regex ]]
#	echo ${BASH_REMATCH[1]}
#done

#list=$(ls -1 ./regularized_fix/*.solverstate | tr '\n' '\0' | xargs -0 -n 1 basename | sort -V)
