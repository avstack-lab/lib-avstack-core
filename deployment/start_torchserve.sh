#!/usr/bin/env bash

set -e

poetry run torchserve --start --ncs \
	--model-store ./model-store \
	--models  fasterrcnn_cityscapes.mar \
	--ts-config ./config.properties
