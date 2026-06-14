<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';

	const app = APPS.find((a) => a.id === 'log')!;

	// Placeholder events. Shape mirrors Event (ts/kind/role/content) so wiring
	// later swaps this for GET /api/events. Newest last, reading order.
	const events = [
		{ id: 1, kind: 'message', role: 'user', content: 'what did I tell you about deploys?' },
		{ id: 2, kind: 'tool_call', role: 'agent', content: 'memory_recall' },
		{ id: 3, kind: 'tool_result', role: 'tool', content: '2 hits: deploys/Fridays, style/short' },
		{ id: 4, kind: 'message', role: 'agent', content: 'You deploy on Fridays. You prefer short answers.' }
	];

	// Per-kind accent so the log scans at a glance.
	const kindColor: Record<string, string> = {
		message: 'text-text',
		tool_call: 'text-glow',
		tool_result: 'text-muted',
		recall: 'text-muted',
		dream: 'text-faint'
	};
</script>

<AppHeader title={app.title}>
	<span class="text-faint text-[10px] lowercase">{events.length}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto text-[11px] leading-relaxed">
	{#each events as e (e.id)}
		<div class="flex gap-3 border-b border-border/40 px-4 py-1.5">
			<span class="text-faint w-6 shrink-0 text-right">{e.id}</span>
			<span class="text-faint w-20 shrink-0">{e.kind}</span>
			<span class="text-faint w-12 shrink-0">{e.role}</span>
			<span class="{kindColor[e.kind] ?? 'text-text'} min-w-0 wrap-break-word">{e.content}</span>
		</div>
	{/each}
</div>
