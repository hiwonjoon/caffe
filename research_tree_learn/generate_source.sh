#!/bin/bash
if ! source ./config.sh
then
	echo "You don't have config sciprt."
	echo "Please make config script."
	exit
fi

echo "Your dataset location is "$TEST_SET_PATH

if ! [ -e "./train_base.txt" -a -e "./test_base.txt" ]
then
	echo "You don't have base file. Please generate base first."
	exit
fi

path=($(realpath $TEST_SET_PATH))
sed -e s#^#$path# ./train_base.txt > train.txt
sed -e s#^#$path# ./test_base.txt > test.txt

