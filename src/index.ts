import { RoleType, type Message, type ToolDef } from "./types.ts";
import { AnthropicClient } from "./bridge/anthropic.ts";
import { runLoop } from "./bridge/loop.ts";
import type { ModelClient } from "./bridge/types.ts";
import { MemoryStore } from "./memory.ts";
import { memoryTools, recallContext } from "./memoryTools.ts";

const BASE_SYSTEM =
    "You are a terse assistant. Use tools when asked about weather. " +
    "Save durable facts and preferences with memory_save, and recall them with memory_recall.";

/** Build the system turn, folding in any memories worth recalling up front. */
function buildSystem(store: MemoryStore): Message {
    const recalled = recallContext(store);
    const text = recalled ? `${BASE_SYSTEM}\n\n${recalled}` : BASE_SYSTEM;
    return {
        sender: { role: RoleType.System },
        timestamp: Date.now(),
        content: [{ kind: "text", text }],
    };
}

const ask: Message = {
    sender: { role: RoleType.User },
    timestamp: Date.now(),
    content: [
        { kind: "text", text: "What's the weather in Dublin? Then tell me in one sentence." },
    ],
};

const weatherTool: ToolDef = {
    name: "get_weather",
    description: "Get the current weather for a city.",
    parameters: {
        type: "object",
        properties: { city: { type: "string", description: "City name" } },
        required: ["city"],
    },
    async run(args) {
        const { city } = args as { city: string };
        return { city, tempC: 14, conditions: "overcast" };
    },
};

async function main() {
    if (!process.env.ANTHROPIC_API_KEY) {
        console.log("harness boot — no ANTHROPIC_API_KEY set, skipping live call.");
        console.log("bridge wired:", new AnthropicClient().provider);
        return;
    }

    const client: ModelClient = new AnthropicClient({ model: process.env.MODEL });
    const store = new MemoryStore(process.env.MEMORY_DB ?? "db.sqlite");

    try {
        // Inject what we already know, and let the model read/write memory.
        const system = buildSystem(store);
        const tools: ToolDef[] = [weatherTool, ...memoryTools(store)];

        // Run the agentic loop: model → tool → model, all in core types.
        const { final, turns } = await runLoop(client, {
            messages: [system, ask],
            tools,
        });

        for (const part of final.message.content) {
            if (part.kind === "text") process.stdout.write(part.text);
        }
        console.log(
            `\n[${final.stopReason}] ${turns} turn(s), ${final.usage.outputTokens} out tokens`,
        );
    } finally {
        store.close();
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
