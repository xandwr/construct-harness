<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';

	const app = APPS.find((a) => a.id === 'memories')!;

	// Placeholder rows. Shape mirrors MemoryView (id/content/tags/importance) so
	// wiring later swaps this for GET /api/memories with no template change.
	const rows = [
		{ id: 3, content: 'Deploys go out on Fridays.', tags: ['ops', 'cadence'], importance: 0.8 },
		{ id: 2, content: 'Prefers short, direct answers.', tags: ['style'], importance: 0.6 },
		{ id: 1, content: 'Working on construct-harness, a TS agent harness.', tags: ['project'], importance: 0.9 }
	];
</script>

<AppHeader title={app.title}>
	<span class="text-faint text-[10px] lowercase">{rows.length}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
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
					<td class="text-muted border-b border-border/40 px-4 py-2 align-top">{m.tags.join(' ')}</td>
					<td class="text-muted border-b border-border/40 px-4 py-2 align-top">{m.importance.toFixed(1)}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
