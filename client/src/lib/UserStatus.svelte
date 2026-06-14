<script lang="ts">
	// The human's presence, the way Discord shows yours at the bottom of the
	// sidebar: a coloured dot and a label for whether you're Online, Away, in Do
	// Not Disturb, or Offline. Online/Away are computed server-side from when you
	// last sent a message (a message ⇒ Online; 15 minutes of silence ⇒ Away); DND
	// and Offline you pin by hand. Clicking opens a small menu to set one.
	//
	// This is the *human's* status, not the Construct's — the signal a daemon that
	// runs while you're away would read to know whether you're around. Today it's
	// cosmetic (the client and daemon launch together, so you boot Online), but the
	// wire it reads from is the one a 24/7 daemon would introspect later.
	//
	// Presentational + self-polling: it owns nothing but the menu's open state and
	// a poll timer, so Away appears after silence without a new message. It lives
	// in the layout, so it's the same on every applet.
	import { onMount } from 'svelte';
	import {
		getPresence,
		setPresence,
		type WirePresence,
		type PresenceState,
		type PresenceChoice
	} from '$lib/api';
	import { agoFromMs } from '$lib/time';

	// How often to recompute presence from the server. Away is a time threshold
	// the server crosses on its own; polling is how the dot notices without a new
	// message. A minute is fine — Away is a 15-minute boundary, not a live cursor.
	const POLL_MS = 60_000;

	let presence = $state<WirePresence | null>(null);
	let open = $state(false);
	let busy = $state(false);

	/** The human label under each state, lowercased to match the interface. */
	const LABELS: Record<PresenceState, string> = {
		online: 'online',
		away: 'away',
		dnd: 'do not disturb',
		offline: 'offline'
	};

	/** The pinnable choices the menu offers, in order. `away` is absent on purpose:
	 *  it's derived from silence, not a status you announce. `online` here means
	 *  "clear any override back to automatic", which reads as Online right away. */
	const CHOICES: { value: PresenceChoice; label: string; note: string }[] = [
		{ value: 'online', label: 'online', note: 'active, following your messages' },
		{ value: 'dnd', label: 'do not disturb', note: "here, but don't interrupt" },
		{ value: 'offline', label: 'offline', note: 'appear away until you message' }
	];

	const current: PresenceState = $derived(presence?.state ?? 'offline');
	const label = $derived(LABELS[current]);

	/** A one-line read under the label: a manual status names itself ("set by you"),
	 *  an automatic one reports how stale the last message is ("active 22m ago"). */
	const detail = $derived.by(() => {
		if (!presence) return '';
		if (presence.manual) return 'set by you';
		if (presence.idleMs === null) return 'no messages yet';
		return `active ${agoFromMs(presence.idleMs)}`;
	});

	async function refresh() {
		try {
			presence = await getPresence();
		} catch {
			// A failed poll leaves the last known state up rather than flickering to
			// an error; the next tick retries. Presence is ambient, never blocking.
		}
	}

	async function choose(value: PresenceChoice) {
		open = false;
		if (busy) return;
		busy = true;
		try {
			presence = await setPresence(value);
		} catch {
			// Pin failed (server down, say): re-read so the dot reflects the truth
			// rather than the click we couldn't land.
			await refresh();
		} finally {
			busy = false;
		}
	}

	onMount(() => {
		refresh();
		const timer = setInterval(refresh, POLL_MS);
		return () => clearInterval(timer);
	});

	/** Dismiss the menu on a click outside it (the backdrop) or Escape. */
	function onKey(e: KeyboardEvent) {
		if (e.key === 'Escape') open = false;
	}
</script>

<svelte:window onkeydown={onKey} />

<div class="relative">
	<!-- The status row: dot + label + detail. Clicking toggles the picker. -->
	<button
		type="button"
		onclick={() => (open = !open)}
		aria-haspopup="menu"
		aria-expanded={open}
		title="set your status"
		class="flex w-full items-center gap-2 px-3 py-2.5 text-left
			text-muted hover:bg-surface/50 hover:text-text"
	>
		<span
			class="status-dot mt-0.5 shrink-0"
			data-state={current}
			style="--dot: var(--color-status-{current})"
		></span>
		<span class="flex min-w-0 flex-col leading-tight">
			<span class="text-text text-xs lowercase">{label}</span>
			{#if detail}
				<span class="text-faint text-[10px] lowercase">{detail}</span>
			{/if}
		</span>
	</button>

	{#if open}
		<!-- A click-catching backdrop so a click anywhere else dismisses the menu. -->
		<button
			type="button"
			class="fixed inset-0 z-10 cursor-default"
			aria-label="close status menu"
			tabindex="-1"
			onclick={() => (open = false)}
		></button>

		<!-- The picker, anchored above the row (it sits at the very bottom of the
		     sidebar). One row per pinnable choice; the current state is marked. -->
		<div
			role="menu"
			aria-label="set your status"
			class="absolute bottom-full left-2 right-2 z-20 mb-1 border border-border bg-surface"
		>
			{#each CHOICES as c (c.value)}
				{@const isCurrent =
					current === c.value ||
					(c.value === 'online' && current === 'away' && !presence?.manual)}
				<button
					type="button"
					role="menuitemradio"
					aria-checked={isCurrent}
					disabled={busy}
					onclick={() => choose(c.value)}
					class="flex w-full items-center gap-2 px-3 py-2 text-left
						{isCurrent ? 'bg-surface-2 text-text' : 'text-muted hover:text-text'}
						disabled:opacity-50"
				>
					<span
						class="status-dot shrink-0"
						data-state={c.value}
						style="--dot: var(--color-status-{c.value})"
					></span>
					<span class="flex min-w-0 flex-col leading-tight">
						<span class="text-xs lowercase">{c.label}</span>
						<span class="text-faint text-[10px] lowercase">{c.note}</span>
					</span>
				</button>
			{/each}
		</div>
	{/if}
</div>
