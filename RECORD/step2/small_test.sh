#!/bin/bash

(
echo "Start1" >> file; \
date >> file; \
sleep 10 >> file; \
date >> file; \
echo "end1" >> file
)&

(
echo "Start2" >> file; \
date >> file; \
sleep 15 >> file; \
date >> file; \
echo "end2" >> file
)&
wait
