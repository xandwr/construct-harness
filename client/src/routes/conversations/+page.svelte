<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { getSessions, ApiError, type SessionSummary } from '$lib/api';
	import { shortWhen } from '$lib/time';

	const app = APPS.find((a) => a.id === 'conversations')!;

	// Live conversation list from GET /api/sessions, newest first. Each row links
	// into chat with ?session=<id>; the live session replays read-write, the rest
	// read-only. The shape (id + when + preview + count) matches the server's
	// derived-from-the-log summary.
	let sessions = $state<SessionSummary[]>([]);
	let error = $state<string | null>(null);
	let loading = $state(true);

	async function load() {
		loading = true;
		error = null;
		try {
			const res = await getSessions();
			sessions = res.sessions;
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load conversations';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		load();
	});
</script>

<AppHeader title={app.title} icon={app.icon}>
	<span class="text-faint text-[10px] lowercase">{loading ? '…' : sessions.length}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
	{#if error}
		<div class="text-faint px-4 py-3 text-xs">{error}</div>
	{:else if !loading && sessions.length === 0}
		<div class="text-faint px-4 py-3 text-xs lowercase">no conversations yet</div>
	{:else}
		{#each sessions as s (s.session)}
			<a
				href="/?session={s.session}"
				class="flex gap-3 border-b border-border/40 px-4 py-3 text-xs hover:bg-surface"
			>
				<span class="text-faint w-24 shrink-0 lowercase">{shortWhen(s.when)}</span>
				<div class="min-w-0 flex-1">
					<div class="text-text truncate">{s.preview || '(no message)'}</div>
					<div class="text-faint mt-0.5 text-[10px]">
						{s.session}{#if s.live}<span class="text-glow"> · live</span>{/if}
					</div>
				</div>
				<span class="text-faint shrink-0 text-[10px]">{s.count}</span>
			</a>
		{/each}
	{/if}
</div>
