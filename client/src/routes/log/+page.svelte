<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { getLog, ApiError, type WireEvent } from '$lib/api';

	const app = APPS.find((a) => a.id === 'log')!;

	// Live event log from GET /api/log. The server returns newest first; we
	// reverse to reading order (newest last) so the log reads top-to-bottom like
	// the conversation it records. Shape mirrors Event (ts/kind/role/content).
	let events = $state<WireEvent[]>([]);
	let total = $state(0);
	let error = $state<string | null>(null);
	let loading = $state(true);

	async function load() {
		loading = true;
		error = null;
		try {
			const res = await getLog();
			events = res.events.slice().reverse();
			total = res.total;
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load the event log';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		load();
	});

	// Per-kind accent so the log scans at a glance.
	const kindColor: Record<string, string> = {
		message: 'text-text',
		tool_call: 'text-glow',
		tool_result: 'text-muted',
		recall: 'text-muted',
		dream: 'text-faint'
	};
</script>

<AppHeader title={app.title} icon={app.icon}>
	<span class="text-faint text-[10px] lowercase">{loading ? '…' : total}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto text-[11px] leading-relaxed">
	{#if error}
		<div class="text-faint px-4 py-3 text-xs">{error}</div>
	{:else if !loading && events.length === 0}
		<div class="text-faint px-4 py-3 text-xs lowercase">log is empty</div>
	{:else}
		{#each events as e (e.id)}
			<div class="flex gap-3 border-b border-border/40 px-4 py-1.5">
				<span class="text-faint w-6 shrink-0 text-right">{e.id}</span>
				<span class="text-faint w-20 shrink-0">{e.kind}</span>
				<span class="text-faint w-12 shrink-0">{e.role ?? ''}</span>
				<span class="{kindColor[e.kind] ?? 'text-text'} min-w-0 wrap-break-word">{e.content}</span>
			</div>
		{/each}
	{/if}
</div>
