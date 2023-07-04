
# torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

FROM appropriate/curl as tini
RUN set -eux; \
  TINI_VERSION=v0.19.0; \
  TINI_ARCH="amd64"; \
  curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
  chmod +x /sbin/tini

FROM python:3.11-slim as torch
WORKDIR /dep
COPY ./torch-requirements.txt /requirements.txt
# pip install torch; pip freeze | grep -v nvidia-cusolver | pip install --no-deps
# RUN pip install -t /dep torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -t /dep -r /requirements.txt --no-deps

FROM python:3.11-slim as nvtx
RUN pip install -t /dep nvtx

FROM python:3.11 as nsight 
RUN curl -L -o repo.deb https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb \
  && dpkg -i ./repo.deb \
  && mv /var/cuda-repo-debian11-12-1-local/nsight-systems-2023.1.2_2023.1.2.43-1_amd64.deb /nsight.deb


FROM python:3.11-slim as deps
WORKDIR /dep
COPY ./other-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps
COPY .cog/tmp/*/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
RUN pip install -t /dep /tmp/cog-0.0.1.dev-py3-none-any.whl --no-deps

FROM python:3.11-slim as model
WORKDIR /src
COPY --from=torch --link /dep/ /src/
RUN pip install -t /src diffusers transformers safetensors
# COPY ./diffusers-requirements.txt /requirements.txt
# RUN pip install -t /src -r /requirements.txt --no-deps # ?
COPY ./version.py ./script/download-weights /src/
RUN python3 download-weights 

FROM python:3.11-slim
COPY --from=tini --link /sbin/tini /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
COPY --from=model --link /src/diffusers-cache /src/diffusers-cache
COPY --from=torch --link /dep/ /src/
COPY --from=deps --link /dep/ /src/
COPY --from=nvtx /dep/ /src/
COPY --from=nsight /nsight.deb /tmp
RUN dpkg -i /tmp/nsight.deb 
COPY ./cog-overwrite/http.py /src/cog/server/http.py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
ENV PATH=$PATH:/usr/local/nvidia/bin
RUN cp /usr/bin/echo /usr/local/bin/pip # prevent k8s from installing anything
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY ./*py ./cog.yaml /src
