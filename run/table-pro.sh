#!/usr/bin/env bash

./eval-gain-pro.py $1 $2 $3 # generator seed longest_length
sort -k 1,3 $1/gain-pro-report.txt | column -s',' -t | less
