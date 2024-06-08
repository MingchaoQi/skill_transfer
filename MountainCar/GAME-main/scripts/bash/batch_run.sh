#!/bin/bash

iterations=1
seconds=600

while getopts i:s: flag
do
    case "${flag}" in
        i) iterations=${OPTARG};;
        e) seconds=${OPTARG};;
    esac
done

for i in {1..$iterations}
do
    sleep 10
    ./keepaway.sh
    sleep $seconds
    ./kill.sh
    sleep 30
done

../rcss-log-extractor/bin/rcssLogExtractor --in logs --out logs