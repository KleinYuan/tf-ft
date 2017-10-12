setup:
	bash setup.sh

train:
	export PYTHONPATH='.'
	python apps/finetune_alexnet_train.py

