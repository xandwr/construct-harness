export const RoleType = {
    Tool: "tool",
    System: "system",
    Agent: "agent",
    User: "user",
} as const;

export type RoleType = (typeof RoleType)[keyof typeof RoleType];

export interface Sender {
    role: RoleType;
    name?: string;
}

export interface TextPart {
    kind: "text";
    text: string;
}

/** An inline image the model can see. `data` is the raw bytes base64-encoded
 *  (no data-URL prefix), `mediaType` the wire MIME the provider expects. Kept
 *  deliberately narrow: only the two formats every provider accepts. */
export interface ImagePart {
    kind: "image";
    mediaType: "image/jpeg" | "image/png";
    data: string; // base64-encoded bytes
}

export interface ToolDef {
    name: string;
    description: string;
    parameters: unknown; // JSON
    run(args: unknown): Promise<unknown>;
}

export interface ToolCallPart {
    kind: "tool_call";
    id: string; // correlates this call with its result
    name: string; // which tool to run
    args: unknown; // parsed JSON arguments
}

export interface ToolResultPart {
    kind: "tool_result";
    callId: string; // must match a ToolCallPart.id
    result: unknown; // tool output
    isError?: boolean;
}

export type ContentPart = TextPart | ImagePart | ToolCallPart | ToolResultPart;

export interface Message {
    sender: Sender;
    timestamp: number;
    content: ContentPart[];
}
