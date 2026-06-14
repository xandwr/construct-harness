<script lang="ts">
	import { tick } from 'svelte';
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { page } from '$app/state';
	import { sendChat, getEvents, ApiError, type ChatEvent, type WireEvent } from '$lib/api';
	import { clock, shortWhen } from '$lib/time';
	import { renderMarkdown } from '$lib/markdown';

	const app = APPS.find((a) => a.id === 'chat')!;

	// A rendered transcript line. `tool` carries a one-line note of tool activity
	// shown under the agent's text; `pending` marks the reply still streaming.
	// `ts` is the epoch-ms send time: the event log's `ts` when replaying, or a
	// client clock reading when the line is created live.
	// `eventId` is the source event's log id when replaying; it anchors the line
	// so the event log can deep-link here with #event-<id>. Live lines have none.
	// `coveredIds` also lists the ids of events folded into this line (the line's
	// own message plus any tool_call events stacked onto it), so the event log can
	// deep-link to a tool event and land on the agent message it belongs to.
	interface Line {
		role: 'user' | 'agent';
		text: string;
		tool?: string;
		// The agent's streamed reasoning trace for this line, accumulated from
		// `thinking` events. Rendered as a collapsible block above the reply;
		// `thinkingOpen` tracks whether the human has expanded it. Live-only:
		// thinking isn't persisted to the log, so replayed lines never have it.
		thinking?: string;
		thinkingOpen?: boolean;
		pending?: boolean;
		ts?: number;
		eventId?: number;
		coveredIds?: number[];
	}

	// When ?session=<id> is present the conversations applet linked here to replay
	// a past conversation read-only (composer disabled). Otherwise this is the live
	// session: the composer is enabled and replies stream in.
	const viewing = $derived(page.url.searchParams.get('session'));

	let messages = $state<Line[]>([]);
	let draft = $state('');
	let sending = $state(false);
	let error = $state<string | null>(null);
	let footer = $state<string | null>(null);
	// The event id the URL hash (#event-<id>) points at: the message the event log
	// linked to. We scroll it into view and flash it once after the replay renders.
	let highlighted = $state<number | null>(null);

	// Reload the transcript whenever the ?session param changes: replay that
	// session's events when viewing one, else start the live view empty (a fresh
	// turn appends to it). Reruns because it reads `viewing`, a $derived.
	$effect(() => {
		const id = viewing;
		error = null;
		footer = null;
		if (!id) {
			messages = [];
			return;
		}
		loadReplay(id);
	});

	// When only the hash changes (same session, different event clicked in the log),
	// the replay effect above doesn't refire, so re-reveal the new target here.
	// Reads page.url.hash to track it; skips while a load is in flight (loadReplay
	// reveals on its own once messages render).
	$effect(() => {
		page.url.hash;
		if (viewing && messages.length) revealHashTarget();
	});

	async function loadReplay(id: string) {
		try {
			const res = await getEvents(id);
			messages = replayLines(res.events);
			await revealHashTarget();
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load conversation';
			messages = [];
		}
	}

	// Scroll the message named by the URL hash (#event-<id>) into view and flash it,
	// so a click in the event log lands the eye on the exact message. The clicked id
	// may be a message (its own line) or a tool_call (the agent line it was folded
	// into), so resolve it to the line that covers it. Runs after the transcript
	// renders (tick) so the anchor element exists. A no-op without a hash.
	async function revealHashTarget() {
		const m = page.url.hash.match(/^#event-(\d+)$/);
		if (!m) {
			highlighted = null;
			return;
		}
		const clicked = Number(m[1]);
		const line = messages.find((l) => l.coveredIds?.includes(clicked));
		if (line?.eventId == null) return;
		await tick();
		const el = document.getElementById(`event-${line.eventId}`);
		if (!el) return;
		el.scrollIntoView({ behavior: 'smooth', block: 'center' });
		highlighted = line.eventId;
	}

	// Fold a session's raw event log into transcript lines: user and agent
	// messages become lines; tool_call events annotate the agent line that
	// follows. This mirrors how the live stream builds the same shape.
	function replayLines(events: WireEvent[]): Line[] {
		const lines: Line[] = [];
		let pendingTool: string | undefined;
		// Ids of events folded into the next agent line (its tool_calls), so the
		// agent line can claim them as deep-link targets alongside its own id.
		let pendingIds: number[] = [];
		for (const e of events) {
			if (e.kind === 'message' && e.role === 'user') {
				lines.push({ role: 'user', text: e.content, ts: e.ts, eventId: e.id, coveredIds: [e.id] });
			} else if (e.kind === 'message' && e.role === 'agent') {
				lines.push({
					role: 'agent',
					text: e.content,
					tool: pendingTool,
					ts: e.ts,
					eventId: e.id,
					coveredIds: [...pendingIds, e.id]
				});
				pendingTool = undefined;
				pendingIds = [];
			} else if (e.kind === 'tool_call') {
				// Stack multiple tool calls in one turn onto a single note.
				pendingTool = pendingTool ? `${pendingTool}, ${e.content}` : e.content;
				pendingIds.push(e.id);
			}
		}
		return lines;
	}

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		const text = draft.trim();
		if (!text || sending || viewing) return;

		draft = '';
		error = null;
		footer = null;
		sending = true;
		messages.push({ role: 'user', text, ts: Date.now() });

		// The agent line we stream into, tracked by its index so every mutation
		// goes through the reactive `messages` proxy (mutating a captured object
		// reference instead would update state Svelte isn't watching, and the
		// reply would never render — even though the turn completed).
		const idx = messages.push({ role: 'agent', text: '', pending: true }) - 1;

		try {
			await sendChat(text, (e: ChatEvent) => onChatEvent(e, idx));
		} catch (e) {
			messages[idx].pending = false;
			error = e instanceof ApiError ? e.message : 'the request failed';
			if (!messages[idx].text) messages[idx].text = `[${error}]`;
		} finally {
			messages[idx].pending = false;
			// Stamp the reply's completion time if no `done` event did (error/abort).
			messages[idx].ts ??= Date.now();
			sending = false;
		}
	}

	function onChatEvent(e: ChatEvent, idx: number) {
		const reply = messages[idx];
		switch (e.kind) {
			case 'text':
				reply.text += e.text;
				break;
			case 'thinking':
				// Accumulate the reasoning trace. Auto-expand while it's the only
				// thing streaming (no reply text yet) so the human sees the model
				// working; they can collapse it, and we don't fight that choice once
				// the answer starts arriving.
				reply.thinking = (reply.thinking ?? '') + e.text;
				if (reply.thinkingOpen === undefined) reply.thinkingOpen = !reply.text;
				break;
			case 'tool': {
				// Note each tool as it starts; mark a failure on its end.
				if (e.phase === 'start') {
					const note = `${e.name}`;
					reply.tool = reply.tool ? `${reply.tool}, ${note}` : note;
				} else if (e.isError) {
					reply.tool = reply.tool ? `${reply.tool} (errored)` : `${e.name} (errored)`;
				}
				break;
			}
			case 'compacted':
				reply.tool = reply.tool ? `${reply.tool}, compacted` : 'compacted';
				break;
			case 'done':
				reply.pending = false;
				reply.ts = Date.now();
				footer =
					`${e.modelTurns} turn(s) · ${e.usage.inputTokens} in / ${e.usage.outputTokens} out` +
					(e.compactions ? ` · ${e.compactions} compaction(s)` : '') +
					(e.stoppedAtMaxTurns ? ' · cut off' : '');
				break;
			case 'error':
				reply.pending = false;
				error = `${e.errorKind}: ${e.message}`;
				if (!reply.text) reply.text = `[${error}]`;
				break;
			case 'open':
				// Session id is available here if we want to deep-link the live
				// turn later; nothing to render for it now.
				break;
		}
	}
</script>

<AppHeader title={app.title} icon={app.icon}>
	{#if viewing}
		<span class="text-faint text-[10px] lowercase">{viewing}</span>
		<a href="/" class="text-muted hover:text-text text-[10px] lowercase underline">live</a>
	{:else if sending}
		<span class="text-glow text-[10px] lowercase">…</span>
	{:else}
		<span class="text-faint text-[10px] lowercase">live</span>
	{/if}
</AppHeader>

<div class="flex min-h-0 flex-1 flex-col">
	<!-- Transcript -->
	<div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
		{#if messages.length === 0 && !error}
			<div class="text-faint text-xs lowercase">
				{viewing ? 'no messages in this conversation' : 'say something to begin'}
			</div>
		{/if}
		{#each messages as m, i (i)}
			<!-- A hairline rule between messages (not above the first) keeps the
			     author boundary legible at wide widths, where text runs far from
			     the left accent bar. The bar marks who; the rule marks where. -->
			<div
				id={m.eventId != null ? `event-${m.eventId}` : undefined}
				class="flex scroll-mt-4 gap-2 border-l-2 py-3 pl-2 transition-colors duration-700 {m.role ===
				'user'
					? 'border-glow'
					: 'border-border'} {i > 0 ? 'border-t border-t-border' : ''} {m.eventId != null &&
				m.eventId === highlighted
					? 'bg-surface'
					: ''}"
			>
				<span class="flex w-12 shrink-0 flex-col text-[10px] leading-tight lowercase">
					<span class="text-faint">{m.role}</span>
					{#if m.ts}
						<span class="text-faint/70 tabular-nums" title={shortWhen(m.ts)}>{clock(m.ts)}</span>
					{/if}
				</span>
				<div class="min-w-0 flex-1">
					{#if m.thinking}
						<!-- Collapsible reasoning trace. The summary toggles it; the body
						     is muted and indented to read as a thought, not the answer. -->
						<div class="mb-1">
							<button
								type="button"
								onclick={() => (m.thinkingOpen = !m.thinkingOpen)}
								class="text-faint hover:text-muted text-[10px] lowercase"
							>
								{m.thinkingOpen ? '▾' : '▸'} thinking
							</button>
							{#if m.thinkingOpen}
								<div
									class="text-faint mt-1 border-l border-border pl-2 text-[10px] leading-relaxed whitespace-pre-wrap italic"
								>
									{m.thinking}
								</div>
							{/if}
						</div>
					{/if}
					{#if m.role === 'agent'}
						<!-- The agent writes markdown; render it as prose. User messages stay
						     literal text (below) so typed markup shows as-typed. The pending
						     cursor lives outside the @html so the streamed source it renders
						     stays clean. -->
						<div class="md text-text text-xs leading-relaxed">
							{@html renderMarkdown(m.text)}{#if m.pending}<span class="text-glow"
									>▍</span
								>{/if}
						</div>
					{:else}
						<span class="text-text text-xs leading-relaxed whitespace-pre-wrap">{m.text}</span>
					{/if}
					{#if m.tool}
						<div class="text-faint mt-1 text-[10px]">{m.tool}</div>
					{/if}
				</div>
			</div>
		{/each}
		{#if error}
			<div class="text-faint text-[10px]">{error}</div>
		{/if}
		{#if footer}
			<div class="text-faint text-[10px]">{footer}</div>
		{/if}
	</div>

	<!-- Composer: disabled while replaying a past conversation. -->
	<form class="border-t border-border px-4 py-3" onsubmit={submit}>
		<div class="flex items-stretch gap-2">
			<input
				bind:value={draft}
				disabled={!!viewing || sending}
				placeholder={viewing ? 'viewing a past conversation' : 'say something'}
				class="placeholder:text-faint flex-1 border border-border bg-surface px-3 py-2 text-xs text-text outline-none focus:border-glow disabled:opacity-50"
			/>
			<button
				type="submit"
				disabled={!!viewing || sending || draft.trim() === ''}
				class="border border-border bg-surface px-4 py-2 text-xs lowercase text-muted disabled:opacity-50"
			>
				{sending ? '…' : 'send'}
			</button>
		</div>
	</form>
</div>

<!-- Markdown prose styling for rendered agent replies. The @html content
     DOMPurify produces isn't seen by Svelte's style scoper, so these are
     :global selectors gated under .md. The aim is terminal-restrained: the
     monospace face and theme palette stay; markdown only adds structure
     (spacing, weight, rules), never new colors or rounded corners. -->
<style>
	/* Vertical rhythm between blocks; the wrapper's own margins are collapsed
	   so a reply doesn't push off its author row. */
	.md :global(> :first-child) {
		margin-top: 0;
	}
	.md :global(> :last-child) {
		margin-bottom: 0;
	}
	.md :global(p),
	.md :global(ul),
	.md :global(ol),
	.md :global(blockquote),
	.md :global(pre),
	.md :global(table) {
		margin: 0.5em 0;
	}

	.md :global(h1),
	.md :global(h2),
	.md :global(h3),
	.md :global(h4) {
		margin: 0.8em 0 0.4em;
		font-weight: 700;
		line-height: 1.3;
	}
	.md :global(h1) {
		font-size: 1.15em;
	}
	.md :global(h2) {
		font-size: 1.08em;
	}
	.md :global(h3),
	.md :global(h4) {
		font-size: 1em;
		color: var(--color-muted);
	}

	.md :global(a) {
		color: var(--color-glow);
		text-decoration: underline;
	}
	.md :global(strong) {
		font-weight: 700;
		color: var(--color-text);
	}
	.md :global(em) {
		font-style: italic;
	}

	.md :global(ul),
	.md :global(ol) {
		padding-left: 1.4em;
	}
	.md :global(li) {
		margin: 0.2em 0;
	}
	.md :global(li::marker) {
		color: var(--color-faint);
	}

	/* Inline code: a faint plate, no rounding (the instrument is square). */
	.md :global(code) {
		background: var(--color-surface-2);
		padding: 0.05em 0.3em;
		color: var(--color-glow);
	}
	/* Fenced blocks: bordered, scrollable, code inside drops the inline plate. */
	.md :global(pre) {
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		padding: 0.6em 0.7em;
		overflow-x: auto;
	}
	.md :global(pre code) {
		background: none;
		padding: 0;
		color: var(--color-text);
	}

	.md :global(blockquote) {
		border-left: 2px solid var(--color-border);
		padding-left: 0.7em;
		color: var(--color-muted);
	}
	.md :global(hr) {
		border: none;
		border-top: 1px solid var(--color-border);
		margin: 0.8em 0;
	}

	.md :global(table) {
		border-collapse: collapse;
		display: block;
		overflow-x: auto;
	}
	.md :global(th),
	.md :global(td) {
		border: 1px solid var(--color-border);
		padding: 0.3em 0.6em;
		text-align: left;
	}
	.md :global(th) {
		color: var(--color-muted);
		font-weight: 700;
	}

	.md :global(img) {
		max-width: 100%;
	}
</style>
