<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { getLog, ApiError, type WireEvent } from '$lib/api';
	import { exactWhen, iso } from '$lib/time';

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

	// The local shell is the one tool that reaches outside the harness, so call it
	// out in the log. Reads the structured meta the loop logs: a tool_call's args
	// (the command) and a tool_result's policy decision (mode, blocked + reason),
	// so a governed-down or refused command is visible at a glance, not buried in
	// the stringified result.
	function shellAudit(e: WireEvent): { label: string; blocked: boolean } | null {
		const meta = e.meta as
			| {
					name?: string;
					args?: { command?: string };
					result?: { policy?: { mode?: string; blocked?: boolean; reason?: string } };
			  }
			| null
			| undefined;
		if (!meta || meta.name !== 'use__user__shell') return null;
		if (e.kind === 'tool_call') {
			const cmd = meta.args?.command;
			return cmd ? { label: `shell · ${cmd}`, blocked: false } : { label: 'shell', blocked: false };
		}
		if (e.kind === 'tool_result') {
			const p = meta.result?.policy;
			if (p?.blocked) return { label: `blocked · ${p.reason ?? 'policy'}`, blocked: true };
			if (p?.mode && p.mode !== 'unrestricted') return { label: `shell · ${p.mode}`, blocked: false };
			return { label: 'shell', blocked: false };
		}
		return null;
	}
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
			{#if e.session}
				<!-- An event tied to a session deep-links into chat: the chat view
				     scrolls to and flashes the matching message (#event-<id>). -->
				<a
					href="/?session={e.session}#event-{e.id}"
					title="open in conversation"
					class="flex gap-3 border-b border-border/40 px-4 py-1.5 hover:bg-surface"
				>
					<span class="text-faint w-6 shrink-0 text-right">{e.id}</span>
					<span class="text-faint w-28 shrink-0 tabular-nums lowercase" title={iso(e.ts)}
						>{exactWhen(e.ts)}</span
					>
					<span class="text-faint w-20 shrink-0">{e.kind}</span>
					<span class="text-faint w-12 shrink-0">{e.role ?? ''}</span>
					<span class="min-w-0 wrap-break-word">
						{#if shellAudit(e)}
							{@const a = shellAudit(e)}
							<span
								class="{a?.blocked ? 'text-glow' : 'text-muted'} font-mono"
								title={a?.blocked ? 'refused by the shell policy' : 'local shell call'}
								>{a?.label}</span
							>
						{:else}
							<span class={kindColor[e.kind] ?? 'text-text'}>{e.content}</span>
						{/if}
					</span>
				</a>
			{:else}
				<!-- No session to link to (e.g. a system event); render it inert. -->
				<div class="flex gap-3 border-b border-border/40 px-4 py-1.5">
					<span class="text-faint w-6 shrink-0 text-right">{e.id}</span>
					<span class="text-faint w-28 shrink-0 tabular-nums lowercase" title={iso(e.ts)}
						>{exactWhen(e.ts)}</span
					>
					<span class="text-faint w-20 shrink-0">{e.kind}</span>
					<span class="text-faint w-12 shrink-0">{e.role ?? ''}</span>
					<span class="min-w-0 wrap-break-word">
						{#if shellAudit(e)}
							{@const a = shellAudit(e)}
							<span
								class="{a?.blocked ? 'text-glow' : 'text-muted'} font-mono"
								title={a?.blocked ? 'refused by the shell policy' : 'local shell call'}
								>{a?.label}</span
							>
						{:else}
							<span class={kindColor[e.kind] ?? 'text-text'}>{e.content}</span>
						{/if}
					</span>
				</div>
			{/if}
		{/each}
	{/if}
</div>
