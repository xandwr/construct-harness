<script lang="ts">
	import { tick } from 'svelte';
	import { goto } from '$app/navigation';
	import AppHeader from '$lib/AppHeader.svelte';
	import Icon from '$lib/Icon.svelte';
	import CommandMenu from '$lib/CommandMenu.svelte';
	import { APPS } from '$lib/apps';
	import { page } from '$app/state';
	import {
		sendChat,
		getEvents,
		getCommands,
		attachmentUrl,
		ApiError,
		type ChatEvent,
		type ChatImage,
		type WireEvent,
		type WireCommand
	} from '$lib/api';
	import { commandPrefix, filterCommands, commandSignature } from '$lib/commands';
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
	// One image rendered in a transcript line. `src` is what an <img> points at:
	// for a replayed message it's the attachment's served URL; for a just-sent
	// live message it's an object URL of the local file, so the picture shows
	// immediately without a round-trip. `alt` is the filename when known.
	interface LineImage {
		src: string;
		alt: string;
	}

	// An image staged in the composer before send. `data` is the base64 the wire
	// carries; `url` is an object URL backing the thumbnail (and reused as the
	// sent message's live <img src>, so we hand ownership to the line on submit
	// rather than revoking it). `name`/`mediaType` label and type it.
	interface PendingImage {
		data: string;
		mediaType: 'image/jpeg' | 'image/png';
		name: string;
		url: string;
	}

	// What the composer accepts. The harness/provider takes only these two.
	const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg'];
	// Mirror of the server's per-image byte cap (MAX_ATTACHMENT_BYTES), so an
	// oversized file is rejected in the browser with a clear message instead of
	// failing the whole turn at the door.
	const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

	interface Line {
		role: 'user' | 'agent';
		text: string;
		images?: LineImage[];
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

	// When ?session=<id> is present the conversations applet linked here to that
	// conversation: we load its transcript and resume it — sending a message picks
	// it back up where it left off (the server rehydrates its history). Without the
	// param this is a fresh conversation; the first turn's `open` event tells us the
	// id the server assigned, which we then track so follow-up turns continue it.
	const viewing = $derived(page.url.searchParams.get('session'));

	let messages = $state<Line[]>([]);
	let draft = $state('');
	// Images staged in the composer, not yet sent. Each carries the base64 the
	// wire wants plus an object URL for its thumbnail (revoked when removed/sent).
	// A turn sends with whatever is staged here; it's cleared on submit.
	let pendingImages = $state<PendingImage[]>([]);
	// The hidden file picker, clicked by the attach button.
	let fileInput = $state<HTMLInputElement | null>(null);
	let sending = $state(false);
	let loadingReplay = $state(false);
	let error = $state<string | null>(null);
	let footer = $state<string | null>(null);
	// The id the composer sends turns into. Equal to `viewing` when resuming a past
	// conversation; for a fresh conversation it's undefined until the first reply's
	// `open` event assigns the server-minted id, after which later turns continue it.
	let activeSession = $state<string | undefined>(undefined);
	// The event id the URL hash (#event-<id>) points at: the message the event log
	// linked to. We scroll it into view and flash it once after the replay renders.
	let highlighted = $state<number | null>(null);
	// A transient status line under the composer for a client-handled command (e.g.
	// `/history`'s count). Distinct from `error`/`footer`; cleared on the next edit.
	let notice = $state<string | null>(null);

	// ── Slash-command menu ──────────────────────────────────────────────────────
	// The catalogue from GET /api/commands, fetched once. The menu filters this as
	// the human types; an empty list (fetch failed or not yet loaded) just means no
	// menu appears, and typed `/text` sends as an ordinary message.
	let commands = $state<WireCommand[]>([]);
	// Index of the keyboard-highlighted row while the menu is open. Reset to 0
	// whenever the visible set changes so the cursor never points past the list.
	let active = $state(0);
	// The human dismissed the menu for the current draft (Escape). Sticky until the
	// draft changes, so Escape closes it without the next keystroke reopening it.
	let dismissed = $state(false);
	// The composer input element, so a Tab/click completion can keep focus there
	// (a menu-button click would otherwise blur the input) and auto-grow can size it.
	let commandInput = $state<HTMLTextAreaElement | null>(null);

	// The text after `/` when the draft is opening a command (single token, no
	// space), else null — see commandPrefix. Drives whether the menu shows at all.
	const prefix = $derived(commandPrefix(draft));
	// The commands matching the current prefix. Empty when the draft isn't a
	// command or nothing matches; the menu renders only when this is non-empty.
	const matches = $derived(prefix === null ? [] : filterCommands(prefix, commands));
	// Show the menu when there's something to show and the human hasn't dismissed
	// it for this draft.
	const menuOpen = $derived(matches.length > 0 && !dismissed);

	$effect(() => {
		getCommands()
			.then((res) => (commands = res.commands))
			.catch(() => {
				/* No menu without a catalogue; typing `/x` just sends as a message. */
			});
	});

	// Keep `active` in range as the filtered set shrinks/grows while typing. Reading
	// matches.length makes this rerun on every filter change.
	$effect(() => {
		if (active >= matches.length) active = 0;
	});

	// The scrollable transcript element, bound below. We read its scroll position to
	// decide whether to show the jump-to-latest chevron and to stick to the bottom
	// while a reply streams.
	let transcript = $state<HTMLDivElement | null>(null);
	// True when the transcript is scrolled (near) the bottom. We hide the chevron
	// there and only auto-stick to new content when the human was already following
	// along at the end — scrolling up to read history shouldn't be yanked back.
	let atBottom = $state(true);

	// How close to the bottom (px) still counts as "at the bottom". A small slack
	// keeps the chevron from flickering on sub-pixel rounding and lets a reply's
	// final lines settle without re-revealing the button.
	const BOTTOM_SLACK = 24;

	function isNearBottom(el: HTMLElement): boolean {
		return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_SLACK;
	}

	function onTranscriptScroll() {
		if (transcript) atBottom = isNearBottom(transcript);
	}

	function scrollToLatest() {
		transcript?.scrollTo({ top: transcript.scrollHeight, behavior: 'smooth' });
	}

	// Keep the view pinned to the newest content as messages grow, but only when the
	// human is already at the bottom — reading back through history stays put. Reads
	// `messages` so it reruns on every streamed chunk and on replay loads; the tick
	// waits for the new content to lay out before measuring the scroll height.
	$effect(() => {
		// Touch both the count and the last line's streamed text so this reruns as
		// new messages arrive and as the live reply grows chunk by chunk.
		messages.length;
		messages.at(-1)?.text;
		if (!atBottom) return;
		tick().then(() => {
			if (transcript) transcript.scrollTop = transcript.scrollHeight;
		});
	});

	// Reload the transcript whenever the ?session param changes: replay that
	// session's events when resuming one, else start a fresh conversation empty.
	// Either way the composer stays enabled — sending resumes the viewed session or
	// starts a new one. Reruns because it reads `viewing`, a $derived.
	$effect(() => {
		const id = viewing;
		error = null;
		footer = null;
		activeSession = id ?? undefined;
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
		loadingReplay = true;
		try {
			const res = await getEvents(id);
			messages = replayLines(res.events);
			await revealHashTarget();
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load conversation';
			messages = [];
		} finally {
			loadingReplay = false;
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
				// Re-attach any images this turn carried, pointing each <img> at the
				// served-bytes URL for its attachment id (the bytes aren't in the event).
				const images = (e.attachments ?? []).map((a) => ({
					src: attachmentUrl(a.id),
					alt: a.filename ?? 'image'
				}));
				lines.push({
					role: 'user',
					text: e.content,
					ts: e.ts,
					eventId: e.id,
					coveredIds: [e.id],
					images: images.length ? images : undefined
				});
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

	// Any edit re-arms the menu: clear a prior dismissal and any command notice so a
	// fresh `/` opens the menu again and stale status lines don't linger.
	function onInput() {
		dismissed = false;
		notice = null;
		autosize();
	}

	// Grow the composer to fit its content, from one row up to a five-row cap, then
	// let it scroll vertically. A textarea won't shrink on its own (its scrollHeight
	// only ever reports the taller of content vs current height), so reset to auto
	// first to remeasure, then clamp. Called on every edit and reactively whenever
	// `draft` is set programmatically (a completion, or clearing it after submit).
	function autosize() {
		const el = commandInput;
		if (!el) return;
		// Reset so scrollHeight reflects content alone, not a previously grown box.
		el.style.height = 'auto';
		// scrollHeight includes the vertical padding, so fold that into the cap too,
		// otherwise the fifth row is clipped by the px-/py-2 padding (border-box).
		const cs = getComputedStyle(el);
		const padding = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
		const max = lineHeightPx * 5 + padding;
		el.style.height = `${Math.min(el.scrollHeight, max)}px`;
		el.style.overflowY = el.scrollHeight > max ? 'auto' : 'hidden';
	}

	// One text row in pixels. Defaults to the textarea's CSS line-height (leading-5 =
	// 20px) and is re-read from the rendered element on mount so the five-row cap
	// tracks the actual font/line-height rather than drifting if that class changes.
	let lineHeightPx = $state(20);

	// Keep the height in sync when `draft` changes by any path other than typing:
	// Tab/menu completion sets it, and submit clears it to ''. Re-running autosize
	// here collapses the box back to one row after a send. The bind also gives us the
	// element on mount, so measure the real line-height once it exists, then size.
	$effect(() => {
		void draft;
		if (commandInput) {
			const lh = parseFloat(getComputedStyle(commandInput).lineHeight);
			if (!Number.isNaN(lh)) lineHeightPx = lh;
		}
		autosize();
	});

	// Drive the menu from the composer's own keystrokes, so the input keeps focus
	// and one cursor serves both mouse and keyboard. Tab completes the highlighted
	// command; the arrows/Escape navigate the menu when it's open.
	//
	// The composer is a textarea (so it wraps instead of overflowing), which means
	// Enter inserts a newline by default. We override that so bare Enter submits the
	// form, where the line is parsed inline (see submit), and Shift+Enter keeps its
	// native behavior of adding a newline for a multi-line message.
	function onKeydown(event: KeyboardEvent) {
		// Bare Enter submits (matching the old <input>); Shift+Enter adds a newline.
		// This holds whether or not the menu is open. Submit parses the line inline,
		// so a typed/Tab-completed command runs on Enter just as it did before.
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			commandInput?.form?.requestSubmit();
			return;
		}
		if (!menuOpen) return;
		switch (event.key) {
			case 'ArrowDown':
				event.preventDefault();
				active = (active + 1) % matches.length;
				break;
			case 'ArrowUp':
				event.preventDefault();
				active = (active - 1 + matches.length) % matches.length;
				break;
			case 'Tab':
				// Complete the highlighted command into the input, like shell tab
				// completion: fill the draft and stop here. The human then edits or
				// hits Enter to run it; Tab on its own runs nothing.
				event.preventDefault();
				completeCommand(matches[active]);
				break;
			case 'Escape':
				event.preventDefault();
				dismissed = true;
				break;
		}
	}

	// Complete a command into the draft (Tab or a menu click): a parameterized one
	// leaves a trailing space so the human types its argument next; a nullary one
	// completes bare. Either way the menu closes (a trailing space closes it; a bare
	// completion is a finished single token the human can now submit) and nothing
	// runs until Enter. Keeps focus in the input so editing/submitting is immediate.
	function completeCommand(cmd: WireCommand) {
		draft = cmd.params.length === 0 ? `/${cmd.name}` : `/${cmd.name} `;
		dismissed = true;
		commandInput?.focus();
	}

	// Run a client-handled slash command, with any inline arguments the submit
	// parser split off (none of today's commands take args; `args` is here so a
	// future parameterized one reads them without rewiring the call site). The
	// catalogue is shared with the REPL, so it lists `/exit` too; here we map the
	// ones a web composer can honor and note the ones it can't, rather than sending
	// any of them to the model as prose.
	function runCommand(cmd: WireCommand, args: string[] = []) {
		void args;
		error = null;
		footer = null;
		switch (cmd.name) {
			case 'reset':
				// A web conversation can't mutate the server's history, so "reset" is
				// starting a fresh one — the same outcome the REPL's /reset gives.
				goto('/');
				notice = 'started a new conversation';
				break;
			case 'history': {
				const count = messages.length;
				notice = `${count} message${count === 1 ? '' : 's'} in this conversation`;
				break;
			}
			case 'help':
				notice = commands.map((c) => commandSignature(c)).join(' · ');
				break;
			default:
				// e.g. /exit — meaningful in the REPL, not in a browser tab.
				notice = `/${cmd.name} isn't available here`;
		}
	}

	// Read one image File into a staged PendingImage: validate its type/size,
	// base64-encode it for the wire, and mint an object URL for its thumbnail.
	// Returns null (with a notice) for anything we won't accept, so a mixed drop
	// of files keeps the good ones. Async because FileReader is.
	async function intakeImage(file: File): Promise<PendingImage | null> {
		const type = file.type === 'image/jpg' ? 'image/jpeg' : file.type;
		if (!ACCEPTED_IMAGE_TYPES.includes(type)) {
			notice = `${file.name || 'that file'} isn't a PNG or JPEG`;
			return null;
		}
		if (file.size > MAX_IMAGE_BYTES) {
			notice = `${file.name || 'image'} is too large (max ${Math.round(MAX_IMAGE_BYTES / 1024 / 1024)} MB)`;
			return null;
		}
		// Read as a data URL, then strip the `data:...;base64,` prefix to the raw
		// base64 the server decodes. A read failure drops just this file.
		const data = await new Promise<string | null>((resolve) => {
			const reader = new FileReader();
			reader.onload = () => {
				const result = typeof reader.result === 'string' ? reader.result : '';
				const comma = result.indexOf(',');
				resolve(comma === -1 ? null : result.slice(comma + 1));
			};
			reader.onerror = () => resolve(null);
			reader.readAsDataURL(file);
		});
		if (!data) {
			notice = `couldn't read ${file.name || 'that image'}`;
			return null;
		}
		return {
			data,
			mediaType: type as 'image/jpeg' | 'image/png',
			name: file.name || 'image',
			url: URL.createObjectURL(file)
		};
	}

	// Stage a batch of files (from the picker, paste, or a future drop), keeping
	// the ones we accept. Clears any prior command notice first so a stale line
	// doesn't read as a result of this attach.
	async function addImages(files: Iterable<File>) {
		notice = null;
		for (const file of files) {
			const img = await intakeImage(file);
			if (img) pendingImages.push(img);
		}
	}

	// The attach button opens the hidden picker; its change handler stages the
	// chosen files, then resets the input so re-picking the same file fires again.
	function onPickFiles(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		if (input.files) addImages(input.files);
		input.value = '';
	}

	// Paste handler on the composer: pull any image files off the clipboard (a
	// screenshot or a copied image lands here as a `file` item) and stage them,
	// the same intake path as the picker. Non-image paste falls through to the
	// textarea's default (text) untouched.
	function onPaste(event: ClipboardEvent) {
		const items = event.clipboardData?.items;
		if (!items) return;
		const files: File[] = [];
		for (const item of items) {
			if (item.kind === 'file' && item.type.startsWith('image/')) {
				const file = item.getAsFile();
				if (file) files.push(file);
			}
		}
		if (files.length === 0) return;
		// We're handling these as attachments, not pasted text.
		event.preventDefault();
		addImages(files);
	}

	// Drop one staged image (the × on its chip), freeing its object URL: it was
	// only ever backing the thumbnail, and this image won't be sent.
	function removePendingImage(index: number) {
		const [removed] = pendingImages.splice(index, 1);
		if (removed) URL.revokeObjectURL(removed.url);
	}

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		const text = draft.trim();

		// Parse a slash command inline: the first token is the command, the rest its
		// arguments. Submitting (Enter / the send button) is the only thing that
		// runs a command — completing one from the menu just fills the input. An
		// unknown `/word` falls through and sends to the model as ordinary prose.
		if (text.startsWith('/')) {
			const [token, ...rest] = text.slice(1).split(/\s+/);
			const word = token.toLowerCase();
			const cmd = commands.find(
				(c) => c.name === word || (c.aliases ?? []).some((a) => a.toLowerCase() === word)
			);
			if (cmd) {
				draft = '';
				runCommand(cmd, rest);
				return;
			}
		}

		// A turn needs text or at least one image. Also block on an in-flight turn
		// or a replay still loading (sending before the transcript is in context
		// would resume from an incomplete history), but not on viewing a past
		// conversation — that's the resume path.
		if ((!text && pendingImages.length === 0) || sending || loadingReplay) return;

		// Take the staged images for this turn and clear the composer's staging.
		// The wire payload carries the base64; the live user line reuses each
		// object URL as its thumbnail src (so the picture shows without a
		// round-trip) and thereby takes ownership of revoking it later.
		const staged = pendingImages;
		pendingImages = [];

		draft = '';
		error = null;
		footer = null;
		notice = null;
		sending = true;
		messages.push({
			role: 'user',
			text,
			ts: Date.now(),
			images: staged.map((img) => ({ src: img.url, alt: img.name }))
		});

		// The agent line we stream into, tracked by its index so every mutation
		// goes through the reactive `messages` proxy (mutating a captured object
		// reference instead would update state Svelte isn't watching, and the
		// reply would never render — even though the turn completed).
		const idx = messages.push({ role: 'agent', text: '', pending: true }) - 1;

		const images: ChatImage[] = staged.map((img) => ({
			mediaType: img.mediaType,
			data: img.data,
			filename: img.name
		}));

		try {
			// Send into the active conversation: the viewed session resumes it; a
			// fresh one (activeSession undefined) gets a new id back via `open`.
			await sendChat(text, (e: ChatEvent) => onChatEvent(e, idx), {
				session: activeSession,
				images
			});
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

	// A one-line note for a tool as it starts. For the local shell (the one tool
	// that reaches outside the harness) show the command it's about to run, so the
	// human sees what's touching their machine rather than a bare tool name.
	function toolNote(name: string, args: unknown): string {
		if (name === 'use__user__shell' && args && typeof args === 'object') {
			const cmd = (args as { command?: unknown }).command;
			if (typeof cmd === 'string' && cmd.trim()) {
				const short = cmd.length > 80 ? cmd.slice(0, 80) + '…' : cmd;
				return `shell: ${short}`;
			}
		}
		return name;
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
					const note = toolNote(e.name, e.args);
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
				// The server tells us which conversation this turn ran under. For a
				// fresh conversation that's a newly-minted id; adopt it so follow-up
				// turns continue the same conversation rather than spawning new ones.
				if (!activeSession) activeSession = e.session;
				break;
		}
	}
</script>

<AppHeader title={app.title} icon={app.icon}>
	{#if activeSession}
		<!-- Resuming (or continuing) a specific conversation: show its id and a way
		     to start a fresh one. -->
		<span class="text-faint text-[10px] lowercase">{activeSession}</span>
		{#if sending}<span class="text-glow text-[10px] lowercase">…</span>{/if}
		<a href="/" class="text-muted hover:text-text text-[10px] lowercase underline">new</a>
	{:else if sending}
		<span class="text-glow text-[10px] lowercase">…</span>
	{:else}
		<span class="text-faint text-[10px] lowercase">new conversation</span>
	{/if}
</AppHeader>

<div class="relative flex min-h-0 flex-1 flex-col">
	<!-- Transcript -->
	<div
		bind:this={transcript}
		onscroll={onTranscriptScroll}
		class="min-h-0 flex-1 overflow-y-auto px-4 py-4"
	>
		{#if messages.length === 0 && !error}
			<div class="text-faint text-xs lowercase">
				{loadingReplay
					? 'loading conversation…'
					: viewing
						? 'no messages here yet — say something to pick it up'
						: 'say something to begin'}
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
						{#if m.text}
							<span class="text-text text-xs leading-relaxed whitespace-pre-wrap">{m.text}</span>
						{/if}
						{#if m.images && m.images.length}
							<!-- Thumbnails the user attached, rendered explicitly (user messages
							     are literal text, not markdown, so an <img> never comes through
							     @html here). Each opens its full image in a new tab. -->
							<div class="mt-1 flex flex-wrap gap-2" class:mt-0={!m.text}>
								{#each m.images as img (img.src)}
									<a href={img.src} target="_blank" rel="noopener noreferrer">
										<img
											src={img.src}
											alt={img.alt}
											title={img.alt}
											class="max-h-40 max-w-48 border border-border object-contain"
										/>
									</a>
								{/each}
							</div>
						{/if}
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

	<!-- Jump to latest: floats over the transcript's bottom edge while the human has
	     scrolled up off the newest message. Clicking smooth-scrolls to the end, which
	     also re-arms the auto-stick effect. Hidden when already at the bottom or with
	     nothing to scroll to. -->
	{#if !atBottom && messages.length > 0}
		<button
			type="button"
			onclick={scrollToLatest}
			title="scroll to latest"
			aria-label="scroll to latest"
			class="absolute bottom-20 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1 border border-border bg-surface px-2 py-1 text-[10px] lowercase text-muted hover:text-text"
		>
			<Icon name="chevron-down" class="size-3.5" />
			latest
		</button>
	{/if}

	<!-- Composer: always enabled (resuming a past conversation is just sending into
	     it); only blocked mid-turn or while a replay is still loading. Opening the
	     draft with `/` pops the command menu above the input (see CommandMenu). -->
	<form class="border-t border-border px-4 py-3" onsubmit={submit}>
		<!-- A one-line result from a client-handled command (e.g. /history's count).
		     Above the input so it reads as a reply to what was just typed. -->
		{#if notice}
			<div class="text-faint mb-2 text-[10px]">{notice}</div>
		{/if}
		<!-- Staged image chips: thumbnails of attachments not yet sent, each with a
		     × to drop it. Sits above the input so it reads as "going out with this
		     message". Only shown while something is staged. -->
		{#if pendingImages.length}
			<div class="mb-2 flex flex-wrap gap-2">
				{#each pendingImages as img, idx (img.url)}
					<div class="relative border border-border">
						<img
							src={img.url}
							alt={img.name}
							title={img.name}
							class="h-14 w-14 object-cover"
						/>
						<button
							type="button"
							onclick={() => removePendingImage(idx)}
							title="remove"
							aria-label="remove image"
							class="bg-surface text-muted hover:text-text absolute -top-2 -right-2 flex size-4 items-center justify-center border border-border text-[10px] leading-none"
						>
							×
						</button>
					</div>
				{/each}
			</div>
		{/if}
		<!-- Hidden picker, opened by the attach button. Accepts PNG/JPEG, multiple. -->
		<input
			bind:this={fileInput}
			type="file"
			accept="image/png,image/jpeg"
			multiple
			onchange={onPickFiles}
			class="hidden"
		/>
		<div class="flex items-stretch gap-2">
			<!-- Attach: opens the hidden picker. Disabled mid-turn / during replay
			     load, matching the input and send button. -->
			<button
				type="button"
				onclick={() => fileInput?.click()}
				disabled={sending || loadingReplay}
				title="attach an image"
				aria-label="attach an image"
				class="text-muted hover:text-text border border-border bg-surface px-3 py-2 text-xs lowercase disabled:opacity-50"
			>
				+ img
			</button>
			<!-- The relative wrapper anchors the floating menu to the input column, so
			     it spans the input's width and sits just above it. -->
			<div class="relative flex-1">
				{#if menuOpen}
					<CommandMenu
						commands={matches}
						{active}
						onselect={completeCommand}
						onhover={(i) => (active = i)}
					/>
				{/if}
				<textarea
					bind:this={commandInput}
					bind:value={draft}
					oninput={onInput}
					onkeydown={onKeydown}
					onpaste={onPaste}
					disabled={sending || loadingReplay}
					rows="1"
					placeholder={loadingReplay
						? 'loading…'
						: viewing
							? 'resume this conversation'
							: 'say something, paste an image, or / for commands'}
					class="placeholder:text-faint block w-full resize-none border border-border bg-surface px-3 py-2 text-xs leading-5 text-text outline-none focus:border-glow disabled:opacity-50"
				></textarea>
			</div>
			<button
				type="submit"
				disabled={sending || loadingReplay || (draft.trim() === '' && pendingImages.length === 0)}
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
