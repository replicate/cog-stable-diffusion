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
	@echo "Loading sd with tensorizer"
	python sd_load_time_tests.py --tensorizer
	@echo "-------------------------"

test_sd_safe_load_time:
	@echo "---------------------------------------------"
	@echo "Testing sd load time on cpu w/safety checker"
	@echo "-------------------------"
	@echo "Loading sd with from pretrained without safetensors"
	python sd_load_time_tests.py --from-pretrained --safety-checker
	@echo "-------------------------"
	@echo "Loading sd with from pretrained with safetensors"
	python sd_load_time_tests.py --from-pretrained --safetensors --safety-checker
	@echo "-------------------------"
	@echo "Loading sd with tensorizer"
	python sd_load_time_tests.py --tensorizer --safety-checker
	@echo "-------------------------"

test_sd_full_load_time:
	@echo "---------------------------------------------"
	@echo "Testing sd load time on gpu w/safety checker"
	@echo "-------------------------"
	@echo "Loading sd with from pretrained without safetensors"
	python sd_load_time_tests.py --from-pretrained --safety-checker --cuda
	@echo "-------------------------"
	@echo "Loading sd with from pretrained with safetensors"
	python sd_load_time_tests.py --from-pretrained --safetensors --safety-checker --cuda
	@echo "-------------------------"
	@echo "Loading sd with tensorizer"
	python sd_load_time_tests.py --tensorizer --safety-checker --cuda
	@echo "-------------------------"

test_sd_unsafe_cuda_time:
	@echo "---------------------------------------------"
	@echo "Testing sd load time on gpu w/safety checker"
	@echo "-------------------------"
	@echo "Loading sd with from pretrained without safetensors"
	python sd_load_time_tests.py --from-pretrained --cuda
	@echo "-------------------------"
	@echo "Loading sd with from pretrained with safetensors"
	python sd_load_time_tests.py --from-pretrained --safetensors --cuda
	@echo "-------------------------"
	@echo "Loading sd with tensorizer"
	python sd_load_time_tests.py --tensorizer --cuda
	@echo "-------------------------"