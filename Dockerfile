FROM node:22-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates && rm -rf /var/lib/apt/lists/*
RUN npm install -g @openai/codex
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONUTF8=1 PYTHONUNBUFFERED=1 CODEX_SANDBOX=read-only
USER node
WORKDIR /home/node/app
COPY --chown=node:node bridge.py start.sh ./
COPY --chown=node:node assistant ./assistant
COPY --chown=node:node migrations ./migrations
EXPOSE 10000
CMD ["sh", "start.sh"]
