import { User } from "./userbase.ts";

export class Message {
    sender: User;
    timestamp: number;
    contents: string;

    constructor(sender: User, contents: string) {
        this.sender = sender;
        this.timestamp = Date.now();
        this.contents = contents;
    }
}

export class Conversation {
    messages: Array<Message>;

    constructor() { this.messages = []; }
    add_message(msg: Message) { this.messages.push(msg); }
}