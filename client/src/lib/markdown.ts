// Render agent message text as markdown, safely.
//
// Agent replies arrive as markdown source and stream in token-by-token, so this
// runs on partial source every frame: `marked` tolerates unclosed fences and
// dangling syntax, rendering what it can and leaving the rest as it fills in.
// The output is then run through DOMPurify before it reaches `@html`, since the
// agent's text is untrusted and a raw `@html` is an XSS sink. We render to a
// flat string (no DOM nodes) so this stays a pure function usable straight in
// markup.
import { browser } from "$app/environment";
import { marked } from "marked";
import DOMPurify from "dompurify";

// `breaks`: treat a single newline as <br>, matching how the transcript read
// before (whitespace-pre-wrap), so a reply's line breaks survive. `gfm`: tables,
// strikethrough, autolinks — the dialect the agent actually writes.
marked.setOptions({ breaks: true, gfm: true });

// DOMPurify needs a real DOM, which only exists in the browser. Its default
// export is only the sanitizer instance there; under SSR it's an uninitialized
// factory with no `.addHook`/`.sanitize`. So hook only in the browser, and let
// renderMarkdown short-circuit during SSR (the transcript is client-only — it
// streams in over fetch/SSE, nothing of it renders server-side).
if (browser) {
    // Open links in a new tab and sever the opener so a rendered link can't
    // reach back into this window. Hooked once at module load.
    DOMPurify.addHook("afterSanitizeAttributes", (node) => {
        if (node.tagName === "A") {
            node.setAttribute("target", "_blank");
            node.setAttribute("rel", "noopener noreferrer");
        }
    });
}

/**
 * Parse markdown `src` to sanitized HTML. Returns a string safe to drop into
 * `@html`. `marked.parse` is synchronous here (no async extensions configured),
 * so the cast is sound; an empty or whitespace-only source yields ''. Returns ''
 * under SSR, since DOMPurify can't sanitize without a DOM and this transcript is
 * hydrated client-side anyway.
 */
export function renderMarkdown(src: string): string {
    if (!src || !browser) return "";
    const html = marked.parse(src) as string;
    return DOMPurify.sanitize(html);
}
