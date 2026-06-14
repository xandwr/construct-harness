<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';

	const app = APPS.find((a) => a.id === 'settings')!;

	// Placeholder config rows. Read-only first cut; wiring later swaps for
	// GET /api/status (model, db path, embedder availability).
	const rows = [
		{ label: 'model', value: 'claude-opus-4-8' },
		{ label: 'memory_db', value: 'db.sqlite' },
		{ label: 'recall', value: 'semantic · embeddings online' },
		{ label: 'compact_at', value: '120000' }
	];
</script>

<AppHeader title={app.title}>
	<span class="text-faint text-[10px] lowercase">read-only</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
	{#each rows as r (r.label)}
		<div class="flex gap-4 border-b border-border/40 px-4 py-2 text-xs">
			<span class="text-faint w-32 shrink-0">{r.label}</span>
			<span class="text-text">{r.value}</span>
		</div>
	{/each}
</div>
