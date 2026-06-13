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

export type ContentPart = TextPart | ToolCallPart | ToolResultPart;

export interface Message {
    sender: Sender;
    timestamp: number;
    content: ContentPart[];
}
