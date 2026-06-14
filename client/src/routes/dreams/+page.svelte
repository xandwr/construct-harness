<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';

	const app = APPS.find((a) => a.id === 'dreams')!;

	// Placeholder dreams. Shape gestures at Dream (persona + scenario + outcome)
	// so wiring later swaps this for GET /api/events?kind=dream.
	const dreams = [
		{
			id: 1,
			persona: 'Mara, staff security engineer',
			scenario: 'A teammate ships an auth change at 5pm Friday.',
			outcome: 'reject'
		},
		{
			id: 2,
			persona: 'Devin, ship-it product lead',
			scenario: 'A perfectionist blocks a low-risk copy fix.',
			outcome: 'pass'
		}
	];
</script>

<AppHeader title={app.title}>
	<span class="text-faint text-[10px] lowercase">{dreams.length}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
	{#each dreams as d (d.id)}
		<div class="flex gap-3 border-b border-border/40 px-4 py-3 text-xs">
			<span class="{d.outcome === 'pass' ? 'text-glow' : 'text-faint'} w-14 shrink-0 lowercase">
				{d.outcome}
			</span>
			<div class="min-w-0">
				<div class="text-muted">{d.persona}</div>
				<div class="text-text mt-0.5">{d.scenario}</div>
			</div>
		</div>
	{/each}
</div>
