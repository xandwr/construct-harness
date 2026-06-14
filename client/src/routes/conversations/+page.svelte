<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';

	const app = APPS.find((a) => a.id === 'conversations')!;

	// Placeholder sessions. Shape mirrors the eventual GET /api/sessions row
	// (session id + first/last timestamp + a preview line + event count), so
	// wiring later swaps this list for the live query. Each row resumes its
	// session by linking into chat with ?session=<id>; chat replays it there.
	// Newest first: most recent conversation on top.
	const sessions = [
		{
			session: 's-2026-06-13-a',
			when: 'today 14:02',
			preview: 'what did I tell you about deploys?',
			count: 12
		},
		{
			session: 's-2026-06-12-c',
			when: 'jun 12 18:40',
			preview: 'walk me through the dream loop outcomes',
			count: 31
		},
		{
			session: 's-2026-06-11-a',
			when: 'jun 11 09:15',
			preview: 'summarize the persona critics from last run',
			count: 8
		}
	];
</script>

<AppHeader title={app.title} icon={app.icon}>
	<span class="text-faint text-[10px] lowercase">{sessions.length}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
	{#each sessions as s (s.session)}
		<a
			href="/?session={s.session}"
			class="flex gap-3 border-b border-border/40 px-4 py-3 text-xs hover:bg-surface"
		>
			<span class="text-faint w-24 shrink-0 lowercase">{s.when}</span>
			<div class="min-w-0 flex-1">
				<div class="text-text truncate">{s.preview}</div>
				<div class="text-faint mt-0.5 text-[10px]">{s.session}</div>
			</div>
			<span class="text-faint shrink-0 text-[10px]">{s.count}</span>
		</a>
	{/each}
</div>
