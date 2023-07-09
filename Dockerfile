FROM appropriate/curl as tini
RUN set -eux; \
  TINI_VERSION=v0.19.0; \
  TINI_ARCH="amd64"; \
  curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
  chmod +x /sbin/tini

# FROM appropriate/curl as pget
# #RUN https://github.com/replicate/pget/releases/download/v0.0.1/pget \
# RUN curl -sSL -o /pget r2-public-worker.drysys.workers.dev/pget \
#   && chmod +x /pget

FROM python:3.10-slim as torch
WORKDIR /dep
COPY ./torch-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps

FROM python:3.10-slim as deps
WORKDIR /dep
COPY ./other-requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt --no-deps
COPY .cog/tmp/*/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
RUN pip install -t /dep /tmp/cog-0.0.1.dev-py3-none-any.whl --no-deps
RUN pip install -t /dep/nya https://r2-public-worker.drysys.workers.dev/nyacomp-0.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# FROM python:3.10-slim as model
# WORKDIR /src
# COPY --from=torch /dep/ /src/
# RUN pip install -t /src diffusers transformers safetensors
# ARG MODEL_FILE="sd-2.1-fp16-safetensors.tar"
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
# # subprocess.run(["tar", "--create", "--file", fname, MODEL_CACHE], shell=True)
# # subprocess.run(["curl", "-v", "-T", fname, "-H", f"Authorization: Bearer {TOKEN}", f"https://storage.googleapis.com/replicate-weights/{fname}"])


FROM appropriate/curl as model
RUN curl -o nya.tar https://r2-public-worker.drysys.workers.dev/nya-sd-8-threads-2023-04-04-.tar
#RUN curl -o stable-hf-cache.tar https://r2-public-worker.drysys.workers.dev/hf-cache-ait-verdant-2022-11-30.tar
# &&
RUN tar -xf nya.tar -C / # /model/nya?
RUN ls /model


FROM python:3.10-slim
COPY --from=tini --link /sbin/tini /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
COPY --from=model /model /src/model
#COPY --from=model --link /src/diffusers-cache /src/diffusers-cache
COPY --from=torch --link /dep/ /src/
COPY --from=deps --link /dep/ /src/
COPY --from=pget --link /pget /usr/bin/pget
COPY --link ./cog-overwrite/http.py /src/cog/server/http.py
COPY --link ./cog-overwrite/predictor.py /src/cog/predictor.py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
ENV PATH=$PATH:/usr/local/nvidia/bin
#nyacomp 
ENV PRELOAD_PATH=/src/model/nya/meta.csv
#RUN sed -i 's:/tmp/::' model/nya/meta.csv
RUN cp /usr/bin/echo /usr/local/bin/pip # prevent k8s from installing anything
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY ./*py ./cog.yaml /src
