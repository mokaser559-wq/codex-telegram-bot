ALTER TABLE bot_state
    ADD COLUMN IF NOT EXISTS last_worker_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_inbox_progress_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_outbox_progress_at TIMESTAMPTZ;

ALTER TABLE inbox
    ADD COLUMN IF NOT EXISTS processing_at TIMESTAMPTZ;

ALTER TABLE outbox
    ADD COLUMN IF NOT EXISTS sending_at TIMESTAMPTZ;

ALTER TABLE inbox ADD COLUMN IF NOT EXISTS generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE outbox ADD COLUMN IF NOT EXISTS generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE outbox DROP CONSTRAINT IF EXISTS outbox_update_id_part_no_key;
CREATE UNIQUE INDEX IF NOT EXISTS outbox_generation_part_idx
    ON outbox (update_id, generation, part_no);
CREATE TABLE IF NOT EXISTS message_revisions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    update_id BIGINT NOT NULL,
    chat_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    generation INTEGER NOT NULL,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE message_revisions ENABLE ROW LEVEL SECURITY;

CREATE TABLE IF NOT EXISTS data_imports (
    import_key TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    row_count BIGINT NOT NULL DEFAULT 0,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE data_imports ENABLE ROW LEVEL SECURITY;
