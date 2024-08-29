JOB_DIR := $(shell pwd)/jobs

# Define the output directory and script path
OUT_DIR := ./llm/output/jobs
SCRIPT_PATH := ./llm/evaluate/gsm8k_dist.py

job-multi-nodes:
		sbatch -v --export=OUT_DIR=$(OUT_DIR),SCRIPT_PATH=$(SCRIPT_PATH) $(JOB_DIR)/multi_nodes.job

job-multi-gpus:
		sbatch -v --export=OUT_DIR=$(OUT_DIR),SCRIPT_PATH=$(SCRIPT_PATH) $(JOB_DIR)/multi_gpus.job
