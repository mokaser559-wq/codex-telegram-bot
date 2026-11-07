CREATE TABLE IF NOT EXISTS bot_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    next_offset BIGINT,
    last_poll_at TIMESTAMPTZ,
    last_poll_ok BOOLEAN NOT NULL DEFAULT FALSE,
    last_update_at TIMESTAMPTZ,
    last_error_code TEXT
);

INSERT INTO bot_state (singleton) VALUES (TRUE) ON CONFLICT (singleton) DO NOTHING;

CREATE TABLE IF NOT EXISTS inbox (
    update_id BIGINT PRIMARY KEY,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'received'
        CHECK (status IN ('received', 'processing', 'retry', 'completed', 'ignored', 'handled_error', 'dead_letter')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_error_code TEXT,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS inbox_ready_idx ON inbox (update_id)
    WHERE status IN ('received', 'retry');

CREATE TABLE IF NOT EXISTS messages (
    update_id BIGINT NOT NULL,
    chat_id BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    route TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (update_id, role)
);

CREATE INDEX IF NOT EXISTS messages_history_idx
    ON messages (chat_id, update_id DESC);

CREATE TABLE IF NOT EXISTS outbox (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    update_id BIGINT NOT NULL REFERENCES inbox(update_id),
    part_no INTEGER NOT NULL CHECK (part_no >= 0),
    chat_id BIGINT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'sending', 'sent', 'dead_letter')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sent_at TIMESTAMPTZ,
    UNIQUE (update_id, part_no)
);

CREATE INDEX IF NOT EXISTS outbox_ready_idx ON outbox (id, chat_id)
    WHERE status = 'pending';

ALTER TABLE bot_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE inbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE outbox ENABLE ROW LEVEL SECURITY;
