FROM node:22-slim AS whisper-build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v1.9.5 https://github.com/ggml-org/whisper.cpp.git /tmp/whisper.cpp \
    && cmake -S /tmp/whisper.cpp -B /tmp/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
    && cmake --build /tmp/whisper.cpp/build --config Release --target whisper-cli -j2 \
    && mkdir -p /opt/whisper/models \
    && curl -fL --retry 3 https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin -o /opt/whisper/models/ggml-tiny-q5_1.bin \
    && echo "818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7  /opt/whisper/models/ggml-tiny-q5_1.bin" | sha256sum -c -

FROM node:22-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=whisper-build /tmp/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=whisper-build /opt/whisper/models/ggml-tiny-q5_1.bin /opt/whisper/models/ggml-tiny-q5_1.bin
RUN npm install -g @openai/codex
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONUTF8=1 PYTHONUNBUFFERED=1 CODEX_SANDBOX=read-only WHISPER_MODEL_PATH=/opt/whisper/models/ggml-tiny-q5_1.bin
USER node
WORKDIR /home/node/app
COPY --chown=node:node bridge.py start.sh ./
COPY --chown=node:node assistant ./assistant
COPY --chown=node:node migrations ./migrations
CMD ["sh", "start.sh"]
