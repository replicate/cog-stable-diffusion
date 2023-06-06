prepare_jesses_way:
	# echo "Preparing model Jesse's way"
	# python prepare_model_jesses_way.py
	cp -r jesses_way/bin jesses_way/safetensors
	echo "Converting to safetensors"
	python convert.py


test_jesses_way:
	echo "Testing model Jesse's way without safetensors"
	python sd_load_time_tests.py --jesses-way
	echo "Testing model Jesse's way with safetensors"
	python sd_load_time_tests.py --jesses-way --safetensors

test_unet_loading:
	echo "Loading unet with from pretrained without safetensors"
	python unet_load_time_tests.py --from-pretrained
	echo "---------------------------------------------"
	Loading unet with from pretrained with safetensors
	python unet_load_time_tests.py --from-pretrained --safetensors



test_sd_cpu_load_time:
	@echo "---------------------------------------------"
	@echo "Testing sd load time on cpu"
	@echo "-------------------------"
	@echo "Loading sd with from pretrained without safetensors"
	python sd_load_time_tests.py --from-pretrained
	@echo "-------------------------"
	@echo "Loading sd with from pretrained with safetensors"
	python sd_load_time_tests.py --from-pretrained --safetensors
	@echo "-------------------------"
