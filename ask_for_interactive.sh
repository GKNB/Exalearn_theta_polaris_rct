#!/bin/bash

#qsub -I -l select=6,walltime=01:00:00 -l filesystems=home:grand:eagle -A CSC249ADCD08 -q debug-scaling
qsub -I -l select=8,walltime=03:00:00 -l filesystems=home:grand -A CSC249ADCD08 -q preemptable
