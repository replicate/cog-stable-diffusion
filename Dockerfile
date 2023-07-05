
# torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

FROM appropriate/curl as tini
RUN set -eux; \
  TINI_VERSION=v0.19.0; \
  TINI_ARCH="amd64"; \
  curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
  chmod +x /sbin/tini

FROM python:3.10-slim as torch
WORKDIR /dep
COPY ./torch-requirements.txt /requirements.txt
# pip install torch; pip freeze | grep -v nvidia-cusolver | pip install --no-deps
# RUN pip install -t /dep torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -t /dep -r /requirements.txt --no-deps


FROM python:3.10-slim as deps
WORKDIR /dep
COPY ./other-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps
COPY .cog/tmp/*/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
RUN pip install -t /dep /tmp/cog-0.0.1.dev-py3-none-any.whl --no-deps

FROM python:3.10-slim as model
WORKDIR /src
COPY --from=torch /dep/ /src/
RUN pip install -t /src diffusers transformers safetensors
# COPY ./diffusers-requirements.txt /requirements.txt
# RUN pip install -t /src -r /requirements.txt --no-deps # ?
COPY ./version.py ./script/download-weights /src/
RUN python3 download-weights 

FROM python:3.10-slim
COPY --from=tini /sbin/tini /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
COPY --from=model /src/diffusers-cache /src/diffusers-cache
COPY --from=torch /dep/ /src/
COPY --from=deps /dep/ /src/
COPY ./cog-overwrite/http.py /src/cog/server/http.py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
ENV PATH=$PATH:/usr/local/nvidia/bin
RUN cp /usr/bin/echo /usr/local/bin/pip # prevent k8s from installing anything
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY ./*py ./cog.yaml /src
