FROM node:22-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN npm install -g @openai/codex
ENV PYTHONUTF8=1 PYTHONUNBUFFERED=1 CODEX_SANDBOX=read-only
USER node
WORKDIR /home/node/app
COPY --chown=node:node bridge.py start.sh ./
CMD ["sh", "start.sh"]
