#!/bin/bash
#declare -a arr=("antelope" "bat" "beaver" "blue+whale" "bobcat" "buffalo" "chihuahua" "chimpanzee" "collie" "cow" "dalmatian" "deer" "dolphin" "elephant" "fox" "german+shepherd" "giant+panda" "giraffe" "gorilla" "grizzly+bear" "hamster" "hippopotamus" "horse" "humpback+whale" "killer+whale" "leopard" "lion" "mole" "moose" "mouse" "otter" "ox" "persian+cat" "pig" "polar+bear" "rabbit" "raccoon" "rat" "rhinoceros" "seal" "sheep" "siamese+cat" "skunk" "spider+monkey" "squirrel" "tiger" "walrus" "weasel" "wolf" "zebra")

declare -a arr=("cow" "giraffe" "hippopotamus" "horse" "moose" "ox" "pig" "rhinoceros" "sheep" "zebra" "buffalo" "antelope" "deer" )
let a=0
let lines=0
for dir in "${arr[@]}"
do
	path=~/testsets/awa/JPEGImages/$dir/*
	for f in $path
	do
		echo $f $a
		lines=$(($lines+1))
	done
	a=$(($a+1))
done > all.txt
cat all.txt | shuf | tee /dev/stdout > shuffle.txt 
#test/train set ratio
z=$((lines * 50 / 100))
split -l $z shuffle.txt
mv xaa train.txt
mv xab test.txt
rm all.txt shuffle.txt
