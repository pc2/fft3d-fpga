#!/bin/bash

LOGFILE=powermeasure.csv

echo "" > $LOGFILE


# Start the benchmark

$@ &

bm_pid=$!
# Start power measurements

while $(kill -0 $bm_pid); do
    echo $(/usr/share/nallatech/520n/bist/utilities/nalla_serial_cardmon/bin/nalla_serial_cardmon | grep "Total board power (W):" | sed -r 's/.*: ([0-9]+)\.([0-9]+).*/\1.\2/g') >> $LOGFILE
done
