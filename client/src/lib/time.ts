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
