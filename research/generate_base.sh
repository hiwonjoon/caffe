#!/bin/bash
if ! source ./config.sh
then
	echo "You don't have config sciprt."
	echo "Please make config script."
	exit
fi

echo "Your dataset location is "$TEST_SET_PATH

if [ -e "./train_base.txt" -a -e "./test_base.txt" ]
then
	read -p "You already have base file. Do you want to regenerate base?(y/n)" answer
	case ${answer:0:1} in
		n|N )
			exit
		;;
	esac
fi

declare -a arr=("mole" "elephant" "rabbit" "bat" "beaver" "hamster" "squirrel" "rat" "mouse" "spider+monkey" "gorilla" "chimpanzee" "seal" "walrus" "killer+whale" "dolphin" "blue+whale" "humpback+whale" "grizzly+bear" "polar+bear" "skunk" "weasel" "otter" "giant+panda" "raccoon" "fox" "wolf" "dalmatian" "chihuahua" "german+shepherd" "collie" "tiger" "leopard" "lion" "bobcat" "persian+cat" "siamese+cat" "rhinoceros" "horse" "zebra" "hippopotamus" "pig" "giraffe" "moose" "deer" "antelope" "sheep" "buffalo" "ox" "cow" )

let a=0
let lines=0
for dir in "${arr[@]}"
do
	path=($(realpath $TEST_SET_PATH))
	for f in $path/$dir/*
	do
		echo ${f#$path} $a
		lines=$(($lines+1))
	done
	a=$(($a+1))
done > all.txt
cat all.txt | shuf | tee /dev/stdout > shuffle.txt 
#test/train set ratio
z=$((lines * 50 / 100))
split -l $z shuffle.txt
mv xaa train_base.txt
mv xab test_base.txt
rm all.txt shuffle.txt
