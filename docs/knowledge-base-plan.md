# Knowledge Base plan

A user-and-agent-harmonious Knowledge Base: a folder of markdown files plus an
index, editable by **both** a human (in an external editor like Obsidian) and the
agent, kept in sync with a queryable store inside the harness.

This document is the scoped plan and an honest cost estimate. It records the two
decisions that drive the whole shape of the work, why this is a real project and
not a few-day view change, and a phased build order with the failure modes that
actually cost time.

## Decisions taken

Two axes were chosen up front, and both landed on the more expensive option:

1. **Storage model: real `.md` folder with two-way sync.** Not a virtual,
   in-app-only view. The files are real on disk, openable and editable in
   Obsidian or any editor; edits there flow back to the agent, and the agent's
   edits flow out to the files.
2. **Data identity: separate `notes` store, linked to memories.** Notes are
   *not* the existing `memory` rows. A new `notes` table reuses the same
   migration / FTS5 / vector machinery, and a link table relates a note to the
   memories (and other notes) it references. The agent opts into reading notes;
   human documentation and agent memory chatter stay separate corpora.

These two choices are why the estimate is **~3–4 focused weeks**, not the ~3–5
days a virtual view over the existing `memory` table would have cost.

## What we reuse (the cheap part, ~3 days)

The repo already contains, and has proven in production, almost everything the
*store* needs. The `notes` substrate is close to a rename-and-copy of
[`MemoryStore`](../src/memory.ts):

- **Migration runner.** The append-only `MIGRATIONS` array and `migrate()` in
  [`src/memory.ts`](../src/memory.ts) (one `user_version` per file, shared across
  sibling stores) take new migrations for `notes` / `notes_fts` / `notes_vec` /
  `note_links` with no structural change. Append only; never edit a published
  migration.
- **FTS5 external-content index + trigger trio.** The exact pattern at
  [`src/memory.ts`](../src/memory.ts) (`memory_fts`, porter stemmer,
  insert/delete/update triggers, backfill) copies verbatim for `notes_fts`.
- **Selective vector index.** `memory_vec` + its cascade-delete and
  content-update-invalidation triggers copy verbatim for `notes_vec`. Vectors are
  written by the application, never by an insert trigger (an embed is a network
  call), same as today.
- **Shared SQLite helpers.** `clampLimit`, `escapeLike`, `toFtsQuery` in
  [`src/sqlite.ts`](../src/sqlite.ts) are store-agnostic and used as-is.
- **Embedding path.** `embedIfPossible` in
  [`src/memoryTools.ts`](../src/memoryTools.ts) is the model for re-embedding a
  note on content change (it swallows `EmbeddingError` so an embedding outage
  degrades recall to lexical rather than breaking a write).
- **Read-table frontend.** The notes list view starts from the table in
  [`client/src/routes/memories/+page.svelte`](../client/src/routes/memories/+page.svelte),
  which already renders a clean wire shape into a sortable table.

So the **store class, its schema, and a read-only list view are mostly free.**
Everything below is the part that is genuinely new.

## What is new (where the weeks actually go)

### 1. The `notes` schema and store (~3 days)

A new migration set and a `NotesStore` modeled on `MemoryStore`.

Table `notes`:

- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `uuid TEXT NOT NULL UNIQUE` — **stable join key** written into each file's
  frontmatter. This is the identity that survives renames/moves in *either*
  direction. Getting this right on day one is non-negotiable; retrofitting a
  stable id after files exist is painful.
- `path TEXT NOT NULL UNIQUE` — relative path within the KB folder
  (`subdir/title.md`); the "folder" structure is just this column.
- `title TEXT NOT NULL`
- `content TEXT NOT NULL` — the markdown body (frontmatter stripped), FTS- and
  vector-indexed exactly like `memory.content`.
- `frontmatter TEXT` — JSON blob of the parsed frontmatter we don't model as
  columns, so a human can add arbitrary keys without a schema change.
- `content_hash TEXT NOT NULL` — hash of `content` (and relevant frontmatter),
  the basis of conflict detection. Cheaper and more reliable than timestamp
  comparison alone.
- `created INTEGER`, `updated INTEGER` — same semantics as memory.

Table `note_links` (the "linked" in "separate store, linked"):

- `from_note INTEGER REFERENCES notes(id) ON DELETE CASCADE`
- `to_memory INTEGER REFERENCES memory(id) ON DELETE SET NULL`
- `to_note INTEGER REFERENCES notes(id) ON DELETE CASCADE`
- `kind TEXT` — e.g. `references`, `derived_from`.
- Exactly one of `to_memory` / `to_note` set per row. Indexed both directions for
  reverse lookup (this is *relations*, deliberately **not** an Obsidian-style
  backlink graph — no transitive backlink computation, no graph view).

`NotesStore` surface mirrors `MemoryStore`: `save` / `update` / `get` / `getByUuid`
/ `getByPath` / `all` / `search` / `searchRelevant` / `semanticSearch` /
`setEmbedding` / `idsMissingEmbedding` / `delete`. Validation/normalization in the
shape of `normalizeInput`.

### 2. The sync engine (the heart of it, ~1.5–2 weeks)

This is the hard problem in the whole request and the reason two-way sync is a
different animal from a read view. **Nothing in the repo does file I/O today** —
this is a new subsystem (`src/notesSync.ts`).

**Filesystem → DB (inbound):**

- A debounced file watcher over the KB folder (`chokidar`, or `fs.watch` with
  manual debounce). Debounce matters: editors fire multiple events per save.
- Parse frontmatter (`gray-matter` or equivalent), split body from metadata.
- Join by `uuid` in frontmatter. No uuid → it's a human-created file; mint one,
  write it back (carefully — see echo loop below), insert a row.
- On content change: recompute `content_hash`, update the row, **re-embed**
  (model: `embedIfPossible`).
- Handle rename/move (path changed, uuid same → update `path`), and delete
  (file gone → delete row, cascade clears links).

**DB → Filesystem (outbound):**

- Serialize a row to `path/title.md` with frontmatter (uuid, title, tags,
  importance, link references).
- **Atomic writes** (write to temp, rename) so a half-written file is never
  observed by the watcher or the editor.
- Triggered when the agent saves/updates a note via the tool or the write API.

**The echo loop — the classic two-way-sync bug.** An outbound write to disk
fires the watcher, which reads the file back and tries to write it to the DB,
which... A real suppression mechanism is required: track in-flight self-writes by
path + expected `content_hash` and ignore the resulting watcher event. This is
the single most error-prone piece; budget test time for it specifically.

**Conflict resolution.** The agent edits a note's row *while* the human has the
file open and saves it. Single-user-local makes this tractable but not free:

- Policy: compare `content_hash`. If both sides changed since the last synced
  hash, it's a conflict.
- Default resolution: **last-write-wins by `updated`, preserving the loser as a
  `title (conflict <timestamp>).md` sidecar** so nothing is silently destroyed.
  (A real 3-way merge is out of scope; the sidecar is the honest cheap answer.)
- This is where two-way-sync projects die. The mitigation is that there is one
  user and one process, so the conflict window is small — but it must still be
  built and tested, not assumed away.

### 3. Write API and concurrency (~3–4 days)

Today every memory route in [`src/server.ts`](../src/server.ts) is a **read-only
GET**; there is no POST/PUT/DELETE anywhere. The KB needs the write half:

- `GET /api/notes` (list, with `q` search) — mirrors `/api/memories`.
- `GET /api/notes/:uuid` — single note with body + links.
- `POST /api/notes`, `PUT /api/notes/:uuid`, `DELETE /api/notes/:uuid` — write
  paths that update the row, re-embed, **and** trigger the outbound file write,
  converging to the same state the watcher would produce (an API write and a file
  save must not race to two different rows).
- All writes funnel through `NotesStore` + the sync engine's outbound path, so
  there is exactly one code path that mutates a note regardless of origin
  (UI button, agent tool, or file save).

**Concurrency story.** One live `Session` per process (see the process/session
model note in [`src/server.ts`](../src/server.ts)) holds the agent. The human can
edit a file or hit the API while the agent is mid-turn deciding to update the same
note. With one writer process the answer is: serialize writes through the store
(SQLite WAL + busy_timeout already give us this), and let `content_hash` conflict
detection catch the genuinely concurrent human-file-vs-agent-row case. Choose
last-write-wins deliberately; do not discover it.

### 4. Agent tools for notes (~2 days)

Mirror [`src/memoryTools.ts`](../src/memoryTools.ts): `note_save`, `note_update`,
`note_recall` (FTS + semantic, same ranking as memory recall), `note_link` (to a
memory or another note). The agent **opts into** reading notes — notes are not
auto-injected the way memories are, keeping human docs out of every turn's context
unless the agent reaches for them. Optionally a `note_recall` pass in
`recallContext` gated behind a flag.

### 5. Frontend KB app (~3–4 days)

A new applet (`client/src/routes/kb/`) following the existing app pattern
(`APPS` registry, `AppHeader`, the memories table as a starting point):

- A folder/tag tree down the left (the "index"; derived from `path` prefixes
  and/or tags — no separate index file to maintain unless we choose to project
  one).
- A note list / table in the middle.
- A markdown view/edit pane on the right, posting to the write API.
- Show links (note → memory / note → note) as a flat related-list, **not** a
  graph (the backlink graph is explicitly out of scope).

## Out of scope (deliberately)

- **Backlink graph / graph view.** The expensive Obsidian feature we are
  skipping by choice. `note_links` stores explicit relations only; no transitive
  backlink computation.
- **Real 3-way merge.** Conflicts resolve last-write-wins with a `.conflict`
  sidecar.
- **Multi-user / multi-process.** Single-user-local is assumed throughout and is
  what makes the conflict window tractable.

## Phased build order

Each phase is independently testable and leaves the system working.

1. **`notes` schema + `NotesStore`** (copy the memory machinery, add `uuid` /
   `path` / `content_hash`). Unit-tested like `memory.test.ts`. — ~3 days
2. **Write API + read views** over `NotesStore`, no files yet. The KB is fully
   usable in-app at this point. — ~3–4 days
3. **Outbound sync (DB → files)** with atomic writes. Agent/API edits now appear
   on disk. — ~3 days
4. **Inbound sync (files → DB)** with the watcher, frontmatter parse, and the
   echo-loop suppression. This is the riskiest phase; test it hardest. — ~1 week
5. **Conflict detection + sidecar resolution.** — ~3 days
6. **Agent tools** (`note_save` / `note_recall` / `note_link`). — ~2 days
7. **Frontend KB applet** (tree + list + edit pane). — ~3–4 days

## Cost summary

| Scope | Estimate |
| --- | --- |
| `notes` store + schema (reuses memory machinery) | ~3 days |
| Write API + concurrency | ~3–4 days |
| **Sync engine (watcher, echo-loop, conflicts)** | **~1.5–2 weeks** |
| Agent tools | ~2 days |
| Frontend KB applet | ~3–4 days |
| **Total** | **~3–4 weeks** |

The single biggest cost driver is **two-way sync**, and within it the **echo-loop
suppression and conflict resolution**. If that ever proves too expensive to land
reliably, the cheap fallback is one-way export (DB → files, read-only on disk),
which removes the entire inbound watcher + conflict surface and drops the total
back toward ~1.5 weeks. That is the pressure-release valve to keep in mind.
