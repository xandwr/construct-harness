<script lang="ts">
	// The slash-command menu that floats above the chat composer while the draft is
	// opening a command (a leading `/`, no space yet). Presentational: the composer
	// owns the input, the filtering, and the keyboard (arrows to move `active`, Tab
	// to complete, Escape to dismiss); this just renders the rows and reports a
	// click or hover. Selecting a row only completes the command into the draft — it
	// never runs it; the human submits (Enter) to execute, parsed inline. One row
	// per command — its signature on the left, its one-line description on the
	// right — so the human reads the call shape and what it does before completing.
	// The active row is highlighted to mirror the keyboard cursor the composer
	// drives.
	import { commandSignature } from '$lib/commands';
	import type { WireCommand } from '$lib/api';

	let {
		commands,
		active,
		onselect,
		onhover
	}: {
		// The filtered commands to show, in order. Empty means no match — the
		// composer hides the menu rather than rendering an empty box.
		commands: WireCommand[];
		// Index of the keyboard-highlighted row, kept in sync with the composer's
		// own cursor so mouse and keyboard agree on what Tab would complete.
		active: number;
		// Complete a command into the draft (a click): the composer fills the input
		// with `/name` (or `/name ` for a parameterized one). Never runs it.
		onselect: (cmd: WireCommand) => void;
		// Move the highlight to a hovered row, so the mouse and the arrow keys share
		// one notion of which row is active.
		onhover: (index: number) => void;
	} = $props();
</script>

<!-- Anchored above the composer (the parent positions this in a relative wrapper).
     Bordered and on the surface like every other panel; capped in height with its
     own scroll so a long catalogue doesn't push the composer off screen. -->
<div
	class="absolute bottom-full left-0 right-0 mb-1 max-h-64 overflow-y-auto border border-border bg-surface"
	role="listbox"
	aria-label="slash commands"
>
	{#each commands as cmd, i (cmd.name)}
		<button
			type="button"
			role="option"
			aria-selected={i === active}
			onclick={() => onselect(cmd)}
			onmousemove={() => onhover(i)}
			class="flex w-full items-baseline gap-3 px-3 py-2 text-left text-xs {i === active
				? 'bg-surface-2 text-text'
				: 'text-muted hover:text-text'}"
		>
			<span class="text-glow shrink-0 tabular-nums">{commandSignature(cmd)}</span>
			<span class="text-faint min-w-0 flex-1 truncate">{cmd.description}</span>
		</button>
	{/each}
</div>
