import { RoleType, type Message } from "./types.ts";

const hello: Message = {
    sender: { role: RoleType.System },
    timestamp: Date.now(),
    content: [{ kind: "text", text: "harness boot" }],
};

console.log(hello.content[0]);
