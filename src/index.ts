import { Conversation, Message } from "./message.ts";
import { User, Group, UserRegistry } from "./userbase.ts";

const ub = new UserRegistry();
const conv = new Conversation();

const a = ub.users[0];
const b = ub.users[1];

console.log(conv);

const t1 = new Message(a, "Hello A!");
conv.add_message(t1);

const t2 = new Message(b, "Hello to you too, B!");
conv.add_message(t2);

console.log(conv);