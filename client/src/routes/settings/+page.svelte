<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import {
		getStatus,
		updateSettings,
		ApiError,
		type WireStatus,
		type SettingsPatch,
		type EffortLevel
	} from '$lib/api';

	const app = APPS.find((a) => a.id === 'settings')!;

	// The settings applet: the live runtime knobs the Construct actually runs on,
	// plus the read-only context around them. The top half is writable — pick a
	// provider, then one of its model variants; set the effort level; toggle the
	// server and local tools. Each change PATCHes /api/settings and re-renders from
	// the response, so what's shown is the runtime as it now is. The bottom half
	// (storage, features, sessions, shell policy) stays a read-only mirror of the
	// process, the same truthful status the page has always shown.
	let status = $state<WireStatus | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(true);
	// A key naming the control mid-write, so we can disable it and show a hint
	// without freezing the whole page. Null when idle.
	let saving = $state<string | null>(null);

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

	// Apply one knob change: PATCH the subset, then adopt the refreshed status the
	// server echoes back (so every control re-reflects the real runtime, including
	// any field the change cascaded into — e.g. switching provider moves the model).
	// `which` names the control for the busy state; an error leaves the prior status
	// in place so nothing flickers to a wrong value.
	async function apply(which: string, patch: SettingsPatch) {
		if (saving) return;
		saving = which;
		error = null;
		try {
			status = await updateSettings(patch);
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to apply change';
		} finally {
			saving = null;
		}
	}

	// The provider currently in use (drives the model dropdown's options). Falls
	// back to the first catalogued provider when the runtime's model isn't served
	// by any known provider (a bare process).
	const currentProvider = $derived.by(() => {
		if (!status) return null;
		return (
			status.providers.find((p) => p.id === status!.provider.id) ?? status.providers[0] ?? null
		);
	});

	// Switching provider: pick that provider's default-ish model (its current one
	// if it has it, else its first variant) and apply the model change — the
	// provider value follows the model server-side.
	function onProvider(id: string) {
		if (!status) return;
		const p = status.providers.find((x) => x.id === id);
		if (!p || !p.models.length) return;
		const target = p.models.find((m) => m.current) ?? p.models[0];
		if (target.id === status.provider.model) return;
		apply('provider', { model: target.id });
	}

	function onModel(id: string) {
		if (!status || id === status.provider.model) return;
		apply('model', { model: id });
	}

	// Effort: the dropdown carries '' for "default" (clear the level), else a level.
	function onEffort(value: string) {
		const level = value === '' ? null : (value as EffortLevel);
		if (level === (status?.effort.current ?? null)) return;
		apply('effort', { effort: level });
	}

	function toggleServerTool(id: string, enabled: boolean) {
		if (!status) return;
		const next = status.serverTools.filter((t) => t.enabled).map((t) => t.id);
		const set = new Set(next);
		if (enabled) set.add(id);
		else set.delete(id);
		apply(`server:${id}`, { serverTools: [...set] });
	}

	function toggleLocalTool(key: string, enabled: boolean) {
		apply(`local:${key}`, { localTools: { [key]: enabled } });
	}

	function onOff(v: boolean): string {
		return v ? 'on' : 'off';
	}

	// The read-only context sections, derived from the live status — the same
	// truthful mirror of the process the page has always shown, now sitting below
	// the writable knobs.
	const readonlySections = $derived.by(() => {
		if (!status) return [];
		return [
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
					{
						label: 'compact_at',
						value: status.compactAt === null ? '—' : String(status.compactAt)
					},
					{
						label: 'embeddings',
						value: status.embeddingConfigured
							? 'configured · semantic recall'
							: 'not configured · lexical only'
					},
					{ label: 'dreams', value: onOff(status.features.dreams) },
					{ label: 'transcript recall', value: onOff(status.features.transcriptRecall) },
					{ label: 'working mind', value: onOff(status.features.workingMind) }
				]
			},
			{
				title: 'shell policy',
				rows: [
					{ label: 'mode', value: status.shellPolicy.mode },
					{
						label: 'cwd roots',
						value: status.shellPolicy.allowedCwdRoots.length
							? status.shellPolicy.allowedCwdRoots.join(' · ')
							: 'unconfined'
					}
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

	// A one-line capability hint for a model variant in the dropdown.
	function ctxHint(ctx: number, out: number): string {
		const fmt = (n: number) => (n >= 1_000_000 ? `${n / 1_000_000}M` : `${n / 1000}K`);
		return `${fmt(ctx)} ctx · ${fmt(out)} out`;
	}
</script>

<AppHeader title={app.title} icon={app.icon}>
	<span class="text-faint text-[10px] lowercase">{saving ? 'saving…' : loading ? '…' : 'live'}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
	{#if error}
		<div class="text-glow border-b border-border/40 px-4 py-2 text-xs">{error}</div>
	{/if}

	{#if loading}
		<div class="text-faint px-4 py-3 text-xs lowercase">…</div>
	{:else if status}
		<!-- provider + model + effort: the live knobs that drive the next turn -->
		<div class="text-faint border-b border-border/40 px-4 py-1 text-[10px] lowercase tracking-wide">
			model <span class="text-faint/70">· live, all conversations</span>
		</div>

		<!-- provider dropdown -->
		<div class="flex items-center gap-4 border-b border-border/40 px-4 py-2 text-xs">
			<span class="text-faint w-36 shrink-0">provider</span>
			<select
				class="text-text min-w-0 flex-1 cursor-pointer border border-border bg-surface px-2 py-1 text-xs outline-none disabled:opacity-50"
				value={currentProvider?.id ?? ''}
				disabled={saving !== null || status.providers.length < 2}
				onchange={(e) => onProvider((e.currentTarget as HTMLSelectElement).value)}
			>
				{#each status.providers as p (p.id)}
					<option value={p.id}>{p.label}</option>
				{/each}
			</select>
		</div>

		<!-- model dropdown, populated from the selected provider's variants -->
		<div class="flex items-center gap-4 border-b border-border/40 px-4 py-2 text-xs">
			<span class="text-faint w-36 shrink-0">model</span>
			<select
				class="text-text min-w-0 flex-1 cursor-pointer border border-border bg-surface px-2 py-1 text-xs outline-none disabled:opacity-50"
				value={status.provider.model}
				disabled={saving !== null}
				onchange={(e) => onModel((e.currentTarget as HTMLSelectElement).value)}
			>
				{#each currentProvider?.models ?? [] as m (m.id)}
					<option value={m.id}>{m.label} · {ctxHint(m.contextWindow, m.maxOutput)}</option>
				{/each}
			</select>
		</div>

		<!-- effort dropdown -->
		<div class="flex items-center gap-4 border-b border-border/40 px-4 py-2 text-xs">
			<span class="text-faint w-36 shrink-0">effort</span>
			<select
				class="text-text min-w-0 flex-1 cursor-pointer border border-border bg-surface px-2 py-1 text-xs outline-none disabled:opacity-50"
				value={status.effort.current ?? ''}
				disabled={saving !== null}
				onchange={(e) => onEffort((e.currentTarget as HTMLSelectElement).value)}
			>
				<option value="">default (high)</option>
				{#each status.effort.levels as level (level)}
					<option value={level}>{level}</option>
				{/each}
			</select>
		</div>

		<!-- server tools: provider-hosted, toggled live for every conversation -->
		<div class="text-faint border-b border-border/40 px-4 py-1 text-[10px] lowercase tracking-wide">
			server tools <span class="text-faint/70">· live, all conversations</span>
		</div>
		{#each status.serverTools as t (t.id)}
			<button
				class="flex w-full items-center gap-3 border-b border-border/40 px-4 py-2 text-left text-xs hover:bg-surface/40 disabled:opacity-50"
				disabled={saving !== null}
				aria-pressed={t.enabled}
				onclick={() => toggleServerTool(t.id, !t.enabled)}
			>
				<span
					class="flex h-3 w-3 shrink-0 items-center justify-center border
						{t.enabled ? 'border-glow bg-glow/70' : 'border-faint bg-transparent'}"
				></span>
				<span class="{t.enabled ? 'text-text' : 'text-muted'} w-28 shrink-0">{t.label}</span>
				<span class="text-faint min-w-0 flex-1 truncate">{t.note}</span>
				<span class="text-faint/70 shrink-0 text-[10px] lowercase">{t.enabled ? 'on' : 'off'}</span>
			</button>
		{/each}

		<!-- local tools: harness-owned, applies to new conversations -->
		<div class="text-faint border-b border-border/40 px-4 py-1 text-[10px] lowercase tracking-wide">
			local tools <span class="text-faint/70">· new conversations</span>
		</div>
		{#each status.localTools as t (t.key)}
			<button
				class="flex w-full items-start gap-3 border-b border-border/40 px-4 py-2 text-left text-xs hover:bg-surface/40 disabled:opacity-50"
				disabled={saving !== null}
				aria-pressed={t.enabled}
				onclick={() => toggleLocalTool(t.key, !t.enabled)}
			>
				<span
					class="mt-1 flex h-3 w-3 shrink-0 items-center justify-center border
						{t.enabled ? 'border-glow bg-glow/70' : 'border-faint bg-transparent'}"
				></span>
				<span class="{t.enabled ? 'text-text' : 'text-muted'} w-28 shrink-0">{t.label}</span>
				<span class="min-w-0 flex-1">
					<span class="text-faint">{t.note}</span>
					<span class="text-faint/60 block truncate text-[10px]">{t.toolNames.join(' · ')}</span>
				</span>
				<span class="text-faint/70 mt-0.5 shrink-0 text-[10px] lowercase">{t.enabled ? 'on' : 'off'}</span>
			</button>
		{/each}

		<!-- read-only context below the knobs -->
		{#each readonlySections as section (section.title)}
			<div
				class="text-faint border-b border-border/40 px-4 py-1 text-[10px] lowercase tracking-wide"
			>
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
