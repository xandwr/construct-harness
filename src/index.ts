import { RoleType, type Message, type ToolDef } from "./types.ts";
import { AnthropicClient } from "./bridge/anthropic.ts";
import { runLoop } from "./bridge/loop.ts";
import type { ModelClient } from "./bridge/types.ts";

const system: Message = {
    sender: { role: RoleType.System },
    timestamp: Date.now(),
    content: [
        { kind: "text", text: "You are a terse assistant. Use tools when asked about weather." },
    ],
};

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

    // Run the agentic loop: model → tool → model, all in core types.
    const { final, turns } = await runLoop(client, {
        messages: [system, ask],
        tools: [weatherTool],
    });

    for (const part of final.message.content) {
        if (part.kind === "text") process.stdout.write(part.text);
    }
    console.log(`\n[${final.stopReason}] ${turns} turn(s), ${final.usage.outputTokens} out tokens`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
