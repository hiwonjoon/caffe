#! /bin/bash
if [ $# -ne 1 ]
then
	echo "Specify the directory which you want."
	exit 1
fi
cd $1
../../build/tools/caffe train -solver ./solver.prototxt -weights ../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0
cd ..
