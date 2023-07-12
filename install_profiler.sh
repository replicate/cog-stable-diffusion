#!/bin/bash
set -o xtrace -o pipefail
apt update
apt install -yy curl kakoune
cd
curl -Lo repo.deb https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb
dpkg -i ./repo.deb
dpkg -i /var/cuda-repo-debian11-12-1-local/nsight-systems-2023.1.2_2023.1.2.43-1_amd64.deb
apt-get install -yy --fix-broken
cd /src
python3 -m pip install -t . https://r2-public-worker.drysys.workers.dev/nvtx-0.2.5-cp311-cp311-linux_x86_64.whl
