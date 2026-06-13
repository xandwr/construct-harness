import { DatabaseSync } from "node:sqlite";

const db = new DatabaseSync("db.sqlite");

db.exec(`
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        created INTEGER NOT NULL,
        tags TEXT,
        importance REAL
    );
`);

export class Memory {
    id: number;
    content: string;
    created: number;
    tags?: string[];
    importance?: number;

    constructor(content: string, tags?: string[], importance?: number) {
        this.id = 0;
        this.content = content;
        this.created = Date.now();
        this.tags = tags;
        this.importance = importance;
    }

    update() {
        let now = Date.now();
        if (now - this.created > 100000) {
            console.log("Outdated memory: ", this.content);
        }
    }
}

interface MemoryRow {
    id: number;
    content: string;
    created: number;
    tags: string | null;
    importance: number | null;
}

function rowToMemory(row: MemoryRow): Memory {
    const m = new Memory(
        row.content,
        row.tags ? (JSON.parse(row.tags) as string[]) : undefined,
        row.importance ?? undefined,
    );
    m.id = row.id;
    m.created = row.created;
    return m;
}

export class MemoryStore {
    private insertStmt = db.prepare(
        `INSERT INTO memory (content, created, tags, importance)
         VALUES (?, ?, ?, ?)`,
    );
    private getStmt = db.prepare(`SELECT * FROM memory WHERE id = ?`);
    private allStmt = db.prepare(`SELECT * FROM memory ORDER BY importance DESC, created DESC`);
    private deleteStmt = db.prepare(`DELETE FROM memory WHERE id = ?`);

    /** Persist a memory, assigning its real id from the database. */
    save(memory: Memory): Memory {
        const result = this.insertStmt.run(
            memory.content,
            memory.created,
            memory.tags ? JSON.stringify(memory.tags) : null,
            memory.importance ?? null,
        );
        memory.id = Number(result.lastInsertRowid);
        return memory;
    }

    get(id: number): Memory | undefined {
        const row = this.getStmt.get(id) as MemoryRow | undefined;
        return row ? rowToMemory(row) : undefined;
    }

    all(): Memory[] {
        const rows = this.allStmt.all() as unknown as MemoryRow[];
        return rows.map(rowToMemory);
    }

    delete(id: number): boolean {
        return this.deleteStmt.run(id).changes > 0;
    }
}
