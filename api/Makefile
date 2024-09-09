all: demo

# https://kylewbanks.com/blog/running-multiple-make-targets-concurrently
demo: 
	make -j 2 server client

server:
	@if [ -z "${MODEL_CHECKPOINT}" ]; then \
		CHECKPOINT='/mnt/nate/model_checkpoints/ref/step3833'; \
	else \
		CHECKPOINT=${MODEL_CHECKPOINT}; \
	fi; \
	export MODEL_CHECKPOINT=$$CHECKPOINT; \
	python worker.py
client:
	GRADIO_TEMP_DIR=./tmpimages python chatbot.py

