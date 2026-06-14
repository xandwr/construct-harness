<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { getStatus, ApiError, type WireStatus } from '$lib/api';

	const app = APPS.find((a) => a.id === 'settings')!;

	// Truthful runtime status: a read-only view of what this process actually is,
	// fetched from GET /api/status. Replaces the old hardcoded rows. No secrets
	// cross the wire — embedding is a yes/no, never a key.
	let status = $state<WireStatus | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(true);

	async function load() {
		loading = true;
		error = null;
		try {
			status = await getStatus();
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load status';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		load();
	});

	function onOff(v: boolean): string {
		return v ? 'on' : 'off';
	}

	// The rows, grouped into sections, derived from the live status. A null/empty
	// value renders as a dash so a missing field reads as "not set", not "0".
	const sections = $derived.by(() => {
		if (!status) return [];
		const caps = Object.entries(status.provider.capabilities)
			.filter(([, on]) => on)
			.map(([k]) => k);
		return [
			{
				title: 'provider',
				rows: [
					{ label: 'model', value: status.provider.model },
					{ label: 'capabilities', value: caps.join(' · ') || '—' },
					{ label: 'server tools', value: status.serverTools.join(' · ') || 'none' }
				]
			},
			{
				title: 'tools',
				rows: [
					{ label: 'local tools', value: status.localTools.join(' · ') || 'none' },
					{ label: 'shell policy', value: status.shellPolicy.mode },
					{
						label: 'shell cwd roots',
						value: status.shellPolicy.allowedCwdRoots.length
							? status.shellPolicy.allowedCwdRoots.join(' · ')
							: 'unconfined'
					}
				]
			},
			{
				title: 'storage',
				rows: [
					{ label: 'memory_db', value: status.storage.memoryDb ?? '—' },
					{ label: 'kb_dir', value: status.storage.kbDir ?? '—' },
					{ label: 'schema version', value: String(status.storage.schemaVersion) },
					{
						label: 'rows',
						value: `${status.storage.memories} memories · ${status.storage.events} events · ${status.storage.goals} goals`
					}
				]
			},
			{
				title: 'context',
				rows: [
					{ label: 'compact_at', value: status.compactAt === null ? '—' : String(status.compactAt) },
					{
						label: 'embeddings',
						value: status.embeddingConfigured ? 'configured · semantic recall' : 'not configured · lexical only'
					},
					{ label: 'dreams', value: onOff(status.features.dreams) },
					{ label: 'transcript recall', value: onOff(status.features.transcriptRecall) },
					{ label: 'working mind', value: onOff(status.features.workingMind) }
				]
			},
			{
				title: 'sessions',
				rows: [
					{
						label: 'live now',
						value: status.liveSessions.length
							? `${status.liveSessions.length} · ${status.liveSessions.map((s) => s.slice(0, 8)).join(', ')}`
							: 'none'
					}
				]
			}
		];
	});
</script>

<AppHeader title={app.title} icon={app.icon}>
	<span class="text-faint text-[10px] lowercase">{loading ? '…' : 'read-only'}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
	{#if error}
		<div class="text-faint px-4 py-3 text-xs">{error}</div>
	{:else if loading}
		<div class="text-faint px-4 py-3 text-xs lowercase">…</div>
	{:else}
		{#each sections as section (section.title)}
			<div class="text-faint border-b border-border/40 px-4 py-1 text-[10px] lowercase tracking-wide">
				{section.title}
			</div>
			{#each section.rows as r (r.label)}
				<div class="flex gap-4 border-b border-border/40 px-4 py-2 text-xs">
					<span class="text-faint w-36 shrink-0">{r.label}</span>
					<span class="text-text break-all">{r.value}</span>
				</div>
			{/each}
		{/each}
	{/if}
</div>
