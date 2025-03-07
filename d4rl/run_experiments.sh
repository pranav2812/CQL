#!/bin/bash

# Script to reproduce results

envs=(
	"halfcheetah-random-v0"
	"halfcheetah-medium-v0"
	"halfcheetah-medium-replay-v0"
	"halfcheetah-medium-expert-v0"
	"halfcheetah-expert-v0"
	"hopper-random-v0"
	"hopper-medium-v0"
	"hopper-medium-replay-v0"
	"hopper-medium-expert-v0"
	"hopper-expert-v0"
	"walker2d-random-v0"
	"walker2d-medium-v0"
	"walker2d-medium-replay-v0"
	"walker2d-medium-expert-v0"
	"walker2d-expert-v0"
	)


for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		python examples/cql_mujoco_new.py --env=$env --policy_lr=0.0001 \
		--seed=$i --lagrange_thresh=-1 \
		--min_q_weight=5 --gpu=0 --min_q_version=3
	done
done

python examples/cql_mujoco_new.py --env=halfcheetah-random-v0 --policy_lr=0.0001 --seed=0 --lagrange_thresh=-1 --min_q_weight=5 --gpu=0 --min_q_version=3