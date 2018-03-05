all:
	./train.sh

clean:
	rm -rf ./log

tensorboard:
	tensorboard --logdir ./log --port 6601

tfrecords:
	./build_image.sh

download:
	./download_data.sh
	./download_pretrained_model.sh

freezing:
	python3 ./freezing.py
