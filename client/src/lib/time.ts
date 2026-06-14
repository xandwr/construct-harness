/**
 * Compact timestamp formatting for the lists. The server sends event times as
 * epoch ms; the UI wants something short and legible ("today 14:02", "jun 12
 * 18:40") rather than a full ISO string. Kept tiny and dependency-free.
 */

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

function pad(n: number): string {
    return n < 10 ? `0${n}` : String(n);
}

/** "today 14:02" for today, else "jun 12 18:40". Lowercase to match the UI. */
export function shortWhen(ms: number): string {
    const d = new Date(ms);
    const now = new Date();
    const time = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    const sameDay =
        d.getFullYear() === now.getFullYear() &&
        d.getMonth() === now.getMonth() &&
        d.getDate() === now.getDate();
    if (sameDay) return `today ${time}`;
    return `${MONTHS[d.getMonth()]} ${d.getDate()} ${time}`;
}

/** Just the clock, "14:02". For inline use where the date is implied (a single
 *  conversation's messages); pair with {@link shortWhen} as a hover title. */
export function clock(ms: number): string {
    const d = new Date(ms);
    return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** The exact event time, seconds-precise: "jun 12 18:40:07". Where {@link clock}
 *  and {@link shortWhen} round to the minute, this is for the event log, which
 *  records the precise moment a thing happened; pair with {@link iso} as a hover
 *  title for the full machine-readable timestamp. */
export function exactWhen(ms: number): string {
    const d = new Date(ms);
    const time = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    return `${MONTHS[d.getMonth()]} ${d.getDate()} ${time}`;
}

/** The full ISO-8601 timestamp, for a hover title or copy: nothing rounded, the
 *  exact millisecond and timezone offset. */
export function iso(ms: number): string {
    return new Date(ms).toISOString();
}
