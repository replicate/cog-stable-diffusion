
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


FROM python:3.11-slim as deps
WORKDIR /dep
COPY ./other-requirements.txt /requirements.txt
# pip install torch; pip freeze | grep -v nvidia-cusolver | pip install --no-deps
# RUN pip install -t /dep torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -t /dep -r /requirements.txt --no-deps

FROM deps as model
WORKDIR /src
COPY --from=torch /dep/ /src/
COPY --from=deps /dep/ /src/
COPY ./version.py ./script/download-weights /src/
RUN python3 download-weights 

FROM python:3.11-slim
COPY --from=tini /sbin/tini /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
COPY --from=model /src/diffusers-cache /src/diffusers-cache
COPY --from=torch /dep/ /src/
COPY --from=deps /dep/ /src/
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY ./*py /src
