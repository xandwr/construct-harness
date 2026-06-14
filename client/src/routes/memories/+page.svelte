<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { getMemories, ApiError, type WireMemory } from '$lib/api';

	const app = APPS.find((a) => a.id === 'memories')!;

	// Live memory store. Loads on mount from GET /api/memories; the row shape
	// matches the server's memory view (id/content/tags/importance), so the table
	// renders the curated store directly.
	let rows = $state<WireMemory[]>([]);
	let total = $state(0);
	let error = $state<string | null>(null);
	let loading = $state(true);

	async function load() {
		loading = true;
		error = null;
		try {
			const res = await getMemories();
			rows = res.memories;
			total = res.total;
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load memories';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		load();
	});
</script>

<AppHeader title={app.title} icon={app.icon}>
	{#if loading}
		<span class="text-faint text-[10px] lowercase">…</span>
	{:else}
		<span class="text-faint text-[10px] lowercase">{total}</span>
	{/if}
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
	{#if error}
		<div class="text-faint px-4 py-3 text-xs">{error}</div>
	{:else if !loading && rows.length === 0}
		<div class="text-faint px-4 py-3 text-xs lowercase">no memories yet</div>
	{:else}
		<table class="w-full border-collapse text-xs">
			<thead>
				<tr class="text-faint text-left text-[10px] lowercase">
					<th class="border-b border-border px-4 py-2 font-normal">id</th>
					<th class="border-b border-border px-4 py-2 font-normal">content</th>
					<th class="border-b border-border px-4 py-2 font-normal">tags</th>
					<th class="border-b border-border px-4 py-2 font-normal">imp</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as m (m.id)}
					<tr class="hover:bg-surface/50">
						<td class="text-faint border-b border-border/40 px-4 py-2 align-top">{m.id}</td>
						<td class="text-text border-b border-border/40 px-4 py-2 align-top">{m.content}</td>
						<td class="text-muted border-b border-border/40 px-4 py-2 align-top"
							>{m.tags.join(' ')}</td
						>
						<td class="text-muted border-b border-border/40 px-4 py-2 align-top"
							>{m.importance === null ? '—' : m.importance.toFixed(1)}</td
						>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
</div>
