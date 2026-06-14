<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { getContext, ApiError, type WireContext } from '$lib/api';
	import { page } from '$app/state';

	const app = APPS.find((a) => a.id === 'context')!;

	// The context inspector: a dev-oriented view of what the Construct actually
	// sees before it answers. We POST nothing to the model — GET /api/context
	// assembles the same ingredients a turn would (base, recalled memory, goals,
	// last dream, working mind) with per-section token estimates and source ids,
	// and is read-only on the server (no reinforce, no working-mind tick).

	// Prefill the session from ?session= so "inspect this conversation" deep-links
	// from elsewhere land here scoped.
	let session = $state(page.url.searchParams.get('session') ?? '');
	let query = $state('');
	let result = $state<WireContext | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(false);

	async function preview() {
		loading = true;
		error = null;
		try {
			result = await getContext({ session: session || undefined, q: query || undefined });
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to assemble context';
		} finally {
			loading = false;
		}
	}

	// Load once on mount so the standing context (goals, last dream, working mind)
	// is visible without typing a draft.
	$effect(() => {
		preview();
	});

	function ids(label: string, list: number[] | undefined): string {
		return list && list.length ? `${label}: ${list.map((n) => `#${n}`).join(' ')}` : '';
	}
</script>

<AppHeader title={app.title} icon={app.icon}>
	{#if result}
		<span class="text-faint text-[10px] lowercase">~{result.totalTokens} tok</span>
	{/if}
</AppHeader>

<div class="flex min-h-0 flex-1 flex-col">
	<!-- Controls: which session, and the draft to recall against. -->
	<div class="flex flex-col gap-2 border-b border-border px-4 py-2">
		<input
			class="text-text placeholder:text-faint w-full bg-transparent text-[11px] outline-none"
			placeholder="session id (blank = default conversation)"
			bind:value={session}
			onkeydown={(e) => e.key === 'Enter' && preview()}
		/>
		<div class="flex items-center gap-2">
			<input
				class="text-text placeholder:text-faint min-w-0 flex-1 bg-transparent text-xs outline-none"
				placeholder="draft message to recall against…"
				bind:value={query}
				onkeydown={(e) => e.key === 'Enter' && preview()}
			/>
			<button
				class="text-glow hover:text-text shrink-0 text-[10px] lowercase disabled:opacity-40"
				disabled={loading}
				onclick={preview}>{loading ? 'assembling…' : 'preview'}</button
			>
		</div>
	</div>

	{#if error}
		<div class="text-faint px-4 py-3 text-xs">{error}</div>
	{:else if result}
		<div class="min-h-0 flex-1 overflow-y-auto">
			<div class="text-faint border-b border-border/40 px-4 py-1 text-[10px] lowercase">
				session {result.session.slice(0, 12)} · {result.sections.length} sections · ~{result.totalTokens}
				tokens
			</div>
			{#each result.sections as s (s.name)}
				<div class="border-b border-border/40 px-4 py-2">
					<div class="mb-1 flex items-center justify-between gap-3">
						<span class="text-glow text-[11px] lowercase tracking-wide">{s.name}</span>
						<span class="text-faint text-[10px]">~{s.tokens} tok</span>
					</div>
					{#if s.memoryIds?.length || s.goalIds?.length || s.dreamId !== undefined}
						<div class="text-faint mb-1 text-[10px]">
							{[
								ids('memories', s.memoryIds),
								ids('goals', s.goalIds),
								s.dreamId !== undefined ? `dream: #${s.dreamId}` : ''
							]
								.filter(Boolean)
								.join(' · ')}
						</div>
					{/if}
					<pre
						class="text-muted whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">{s.text}</pre>
				</div>
			{/each}
		</div>
	{/if}
</div>
