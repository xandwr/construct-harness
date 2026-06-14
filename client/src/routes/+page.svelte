<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { page } from '$app/state';

	const app = APPS.find((a) => a.id === 'chat')!;

	// chat shows the live session by default, but the conversations applet links
	// here with ?session=<id> to replay a past one. When that param is present we
	// render that session read-only (composer disabled) with a way back to live.
	// Wiring later swaps the placeholder for GET /api/events?session=<id>.
	const viewing = $derived(page.url.searchParams.get('session'));

	// Placeholder transcript. Replaced by the live SSE stream once chat is wired.
	// When viewing a past session it stands in for that session's replayed events.
	const messages = [
		{ role: 'user', text: 'what did I tell you about deploys?' },
		{
			role: 'agent',
			text: 'You deploy on Fridays. You prefer short answers.',
			tool: 'recall deploys → 2'
		}
	];
</script>

<AppHeader title={app.title} icon={app.icon}>
	{#if viewing}
		<span class="text-faint text-[10px] lowercase">{viewing}</span>
		<a href="/" class="text-muted hover:text-text text-[10px] lowercase underline">live</a>
	{:else}
		<span class="text-faint text-[10px] lowercase">not wired</span>
	{/if}
</AppHeader>

<div class="flex min-h-0 flex-1 flex-col">
	<!-- Transcript -->
	<div class="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-4">
		{#each messages as m, i (i)}
			<div class="flex gap-2">
				<span class="text-faint w-12 shrink-0 text-[10px] lowercase">{m.role}</span>
				<div class="min-w-0 flex-1">
					<span class="text-text text-xs leading-relaxed">{m.text}</span>
					{#if m.tool}
						<div class="text-faint mt-1 text-[10px]">{m.tool}</div>
					{/if}
				</div>
			</div>
		{/each}
	</div>

	<!-- Composer (disabled until wiring) -->
	<form class="border-t border-border px-4 py-3" onsubmit={(e) => e.preventDefault()}>
		<div class="flex items-stretch gap-2">
			<input
				disabled
				placeholder={viewing ? 'viewing a past conversation' : 'say something'}
				class="placeholder:text-faint flex-1 border border-border bg-surface px-3 py-2 text-xs text-text outline-none focus:border-glow disabled:opacity-50"
			/>
			<button
				type="submit"
				disabled
				class="border border-border bg-surface px-4 py-2 text-xs lowercase text-muted disabled:opacity-50"
			>
				send
			</button>
		</div>
	</form>
</div>
