FROM nvidia/cuda:11.8.0-devel-ubuntu18.04 as cuda

# seperate tini stage to not need curl in final stage
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

# torch dependencies, except for torch itself
FROM python:3.11-slim as torch-deps
WORKDIR /dep
COPY ./torch-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps

FROM appropriate/curl as torch
# it's not really necessary to use the curl image here 
WORKDIR /dep
# this is torch compiled with https://github.com/technillogue/build-pytorch/blob/main/build-pytorch/Dockerfile
# unlike the version from pypi, this is not statically compiled bundled manywheel; it needs cuda libs separately 
COPY torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl /dep
RUN unzip torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl
RUN apk update && apk add patchelf && patchelf --remove-needed libcurand.so.10 torch/lib/libtorch_cuda.so && patchelf --remove-needed libcurand.so.10 torch/lib/libtorch_global_deps.so

FROM python:3.11-slim as deps
# install other dependencies into /dep. subdependencies are already resolved
WORKDIR /dep
COPY ./other-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps
# don't bother installing the version of cog-python bundled with cog-go 
RUN pip install -t /dep cog==0.8.1 --no-deps


FROM python:3.11-slim
COPY --from=tini --link /sbin/tini /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
RUN mkdir -p /usr/local/cuda/lib64
# because we compiled torch, we need these cuda libraries
# we copy only the ones we need from the huge base image 
# possibly only the first 3-4 are necessary in principle 
COPY --from=cuda --link \
 /usr/local/cuda/lib64/libcublas.so.11 \
 /usr/local/cuda/lib64/libcublasLt.so.11 \
 /usr/local/cuda/lib64/libcudart.so.11.0 \
 /usr/local/cuda/lib64/libnvToolsExt.so.1 \
 /usr/local/cuda/lib64/libcufft.so.10 \
 /usr/local/cuda/lib64/libcusparse.so.11 \
 # /usr/local/cuda/lib64/libnvrtc.so* \
 # /usr/local/cuda/lib64/libcupti.so.11.8 \
 # /usr/local/cuda/lib64/libcurand.so.10 \
 /usr/local/cuda/lib64

COPY --from=cuda --link /usr/lib/x86_64-linux-gnu/libgomp.so.1* /usr/lib/x86_64-linux-gnu
COPY --from=torch --link /dep/torch/ /src/torch/
COPY --from=torch --link /dep/torch-2.0.0a0+gite9ebda2.dist-info/ /src/torch-2.0.0a0+gite9ebda2.dist-info/
COPY --from=torch-deps --link /dep/ /src/
COPY --from=deps --link /dep/ /src/
COPY --from=pget --link /pget /usr/bin/pget
# patch over cog to avoid reimporting predict, defer imports, logging, etc
COPY --link ./cog-overwrite/ /src/cog/

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/cuda/lib64
ENV PATH=$PATH:/usr/local/nvidia/bin
# this is the file we download with pget
ARG MODEL_FILE="sd-2.1-fp16.pth"
ENV MODEL_FILE=$MODEL_FILE
# prevent k8s from installing anything
RUN ln -s --force /usr/bin/echo /usr/local/bin/pip 
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY ./*sh ./*py ./cog.yaml /src
