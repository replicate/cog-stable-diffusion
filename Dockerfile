FROM nvidia/cuda:11.8.0-devel-ubuntu18.04 as cuda

FROM appropriate/curl as tini
ARG SOURCE_DATE_EPOCH=0
RUN set -eux; \
  TINI_VERSION=v0.19.0; \
  TINI_ARCH="amd64"; \
  curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
  chmod +x /sbin/tini; \
  touch --date="@${SOURCE_DATE_EPOCH}" /sbin/tini

FROM appropriate/curl as pget
ARG SOURCE_DATE_EPOCH=0
#RUN https://github.com/replicate/pget/releases/download/v0.0.1/pget \
RUN curl -sSL -o /pget r2-public-worker.drysys.workers.dev/pget \
  && chmod +x /pget \
  && touch --date="@${SOURCE_DATE_EPOCH}" /pget

FROM python:3.11-slim as torch-deps
WORKDIR /dep
COPY ./torch-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps

FROM appropriate/curl as torch
WORKDIR /dep
COPY torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl /dep
RUN unzip torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl
RUN apk update && apk add patchelf && patchelf --remove-needed libcurand.so.10 torch/lib/libtorch_cuda.so && patchelf --remove-needed libcurand.so.10 torch/lib/libtorch_global_deps.so

FROM python:3.11-slim as deps
WORKDIR /dep
COPY ./other-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps
COPY .cog/tmp/*/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
RUN pip install -t /dep /tmp/cog-0.0.1.dev-py3-none-any.whl --no-deps

# FROM python:3.11 as model
# WORKDIR /src
# COPY --from=torch /dep/ /src/
# RUN pip install -t /src diffusers transformers safetensors
# ARG MODEL_FILE="sd-2.1-fp16.pth"
# ARG GCP_TOKEN # patched 
# ENV MODEL_FILE=$MODEL_FILE
# # you need (gcloud auth print-access-token)
# ENV GCP_TOKEN=$GCP_TOKEN
# # COPY ./diffusers-requirements.txt /requirements.txt
# # RUN pip install -t /src -r /requirements.txt --no-deps # ?
# COPY ./version.py ./script/download-weights /src/
# RUN python3 download-weights && touch /tmp/build
# RUN tar --create --file $MODEL_FILE --directory diffusers-cache . \
#   && curl -vT $MODEL_FILE -H "Authorization: Bearer $GCP_TOKEN" \
#   "https://storage.googleapis.com/replicate-weights/$MODEL_FILE"
# subprocess.run(["tar", "--create", "--file", fname, MODEL_CACHE], shell=True)
# subprocess.run(["curl", "-v", "-T", fname, "-H", f"Authorization: Bearer {TOKEN}", f"https://storage.googleapis.com/replicate-weights/{fname}"])

FROM python:3.11-slim
COPY --from=tini --link /sbin/tini /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]

# for torch compiled from v2.0.1 tag like we did, we need to provide
# libgomp, libcupti, libcudart, libcudart, libcusparse, libcurand, libnvToolsExt, libcufft, libcublas, libcublasLt, libcudart

   # echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
# RUN apt update && apt install -y --no-install-recommends gnupg2 curl ca-certificates \
#     && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | apt-key add - \
#     && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
#     && apt update 
# && apt install -y --no-install-recommends \
#         cuda-cudart-11.8 \
#         cuda-nvrtc-11.8 \
#         libcublas-11.8 \
#         libcurand-11.8 \
#         libcusparse-11.8 \
#         cuda-compat-11.8 \
#         libgomp1 \
#         # libcufft-11.8 \
#         # libcusolver-11.8 \
#         # cuda-nvtx-11.8 \
#     && ln -s cuda-11.8 /usr/local/cuda \
#     && apt-get purge --autoremove -y curl \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* .cache/ \
#     && rm /usr/local/cuda/targets/x86_64-linux/lib/libcusolverMg.so* \
#     && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
#     && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
#COPY --from=model --link /src/diffusers-cache /src/diffusers-cache
RUN mkdir -p /usr/local/cuda/lib64
# only the first 3-4 are necessary in principle 
COPY --from=cuda --link \
 /usr/local/cuda/lib64/libcublas.so.11 \
 /usr/local/cuda/lib64/libcublasLt.so.11 \
 /usr/local/cuda/lib64/libcudart.so.11.0 \
 /usr/local/cuda/lib64/libnvToolsExt.so.1 \
 /usr/local/cuda/lib64/libnvrtc.so* \
 /usr/local/cuda/lib64/libcufft.so.10 \
 /usr/local/cuda/lib64/libcupti.so.11.8 \
 # /usr/local/cuda/lib64/libcurand.so.10 \
 /usr/local/cuda/lib64/libcusparse.so.11 \
 /usr/local/cuda/lib64
COPY --from=cuda --link /usr/lib/x86_64-linux-gnu/libgomp.so.1* /usr/lib/x86_64-linux-gnu
COPY --from=torch --link /dep/torch/ /src/torch/
COPY --from=torch --link /dep/torch-2.0.0a0+gite9ebda2.dist-info/ /src/torch-2.0.0a0+gite9ebda2.dist-info/
COPY --from=torch-deps --link /dep/ /src/
COPY --from=deps --link /dep/ /src/
COPY --from=pget --link /pget /usr/bin/pget
COPY --link ./cog-overwrite/http.py /src/cog/server/http.py
COPY --link ./cog-overwrite/predictor.py /src/cog/predictor.py

#COPY --from=model --link /tmp/build /tmp/build
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/cuda/lib64
ENV PATH=$PATH:/usr/local/nvidia/bin
ARG MODEL_FILE="sd-2.1-fp16.pth"
ARG GCP_TOKEN # patched 
ENV MODEL_FILE=$MODEL_FILE
RUN ln -s --force /usr/bin/echo /usr/local/bin/pip # prevent k8s from installing anything
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY ./*sh ./*py ./cog.yaml /src
