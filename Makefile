NAME?=speech-separation
GPUS?=0
NOTEBOOKS?=/hdd/aidar/notebooks
SAVE_DIR?=/hdd/aidar/ss_save_dir
DATA_DIR?=/hdd/aidar/ss_data_dir
USER?=$(shell whoami)
UID?=$(shell id -u)
GID?=$(shell id -g)

.PHONY: build run

build:
	docker build -t $(NAME) .

run:
	docker run --rm -it --runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=$(GPUS) \
	--ipc=host \
	--net=host \
	-v $(PWD):/workspace \
	-v $(SAVE_DIR):/workspace/saved \
	-v $(NOTEBOOKS):/workspace/notebooks \
	-v $(DATA_DIR):/workspace/data \
	--name=$(NAME) \
	$(NAME) \
	bash
