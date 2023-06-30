
# torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

FROM appropriate/curl as tini
RUN set -eux; \
  TINI_VERSION=v0.19.0; \
  TINI_ARCH="$(dpkg --print-architecture)"; \
  curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
  chmod +x /sbin/tini

RUN pip install -t /dep torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# pip install torch; pip freeze | grep -v nvidia-cusolver | pip install --no-deps
FROM python:3.11-slim as torch
WORKDIR /dep
RUN pip install torch

FROM torch as deps
WORKDIR /dep
RUN find /deps -type f > /tmp/already_installed.txt
COPY .cog/tmp/*/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
RUN pip install -t /dep /tmp/cog-0.0.1.dev-py3-none-any.whl
COPY ./requirements.txt /requirements.txt
RUN pip install -t /dep -r /requirements.txt && cat /tmp/already_installed.txt | xargs -L20 rm -f #mtime?

FROM deps as model
WORKDIR /src
COPY ./version.py ./script/download-weights /src/
RUN python3 download-weights 

FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
COPY --from=tini /sbin/tini /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
COPY --from=model /src/diffusers-cache /src/diffusers-cache
COPY --from=torch /dep/ /src/
COPY --from=deps /dep/ /src/ # hopefully no dups
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY ./*py /src
