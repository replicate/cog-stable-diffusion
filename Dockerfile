ARG MODEL_FILE="sd-2.1-fp16-safetensors.tar"
ARG GCP_TOKEN

FROM appropriate/curl as tini
RUN set -eux; \
  TINI_VERSION=v0.19.0; \
  TINI_ARCH="amd64"; \
  curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
  chmod +x /sbin/tini

FROM appropriate/curl as pget 
RUN curl -sSL -o /pget https://github.com/replicate/pget/releases/download/v0.0.1/pget \
  && chmod +x /pget
FROM python:3.11-slim as torch
WORKDIR /dep
COPY ./torch-requirements.txt /requirements.txt
# pip install torch; pip freeze | grep -v nvidia-cusolver | pip install --no-deps
# RUN pip install -t /dep torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -t /dep -r /requirements.txt --no-deps

FROM python:3.11-slim as deps
WORKDIR /dep
COPY ./other-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps
COPY .cog/tmp/*/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
RUN pip install -t /dep /tmp/cog-0.0.1.dev-py3-none-any.whl --no-deps

FROM python:3.11 as model
WORKDIR /src
COPY --from=torch /dep/ /src/
RUN pip install -t /src diffusers transformers safetensors
ENV MODEL_FILE=$MODEL_FILE
ENV GCP_TOKEN=$GCP_TOKEN
# COPY ./diffusers-requirements.txt /requirements.txt
# RUN pip install -t /src -r /requirements.txt --no-deps # ?
COPY ./version.py ./script/download-weights /src/
RUN python3 download-weights && touch /tmp/build

# upload to GCS / elsewhere

FROM python:3.11-slim
COPY --from=tini --link /sbin/tini /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
#COPY --from=model --link /src/diffusers-cache /src/diffusers-cache
COPY --from=torch --link /dep/ /src/
COPY --from=deps --link /dep/ /src/
COPY --from/pget --link /pget /bin/pget
COPY --link ./cog-overwrite/http.py /src/cog/server/http.py
COPY --link ./cog-overwrite/predictor.py /src/cog/predictor.py
COPY --from=model --link /tmp/build /tmp/build
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
ENV PATH=$PATH:/usr/local/nvidia/bin
ENV MODEL_FILE=$MODEL_FILE
RUN cp /usr/bin/echo /usr/local/bin/pip # prevent k8s from installing anything
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY ./*py ./cog.yaml /src
