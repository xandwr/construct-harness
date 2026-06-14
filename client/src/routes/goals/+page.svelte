<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import {
		getGoals,
		createGoal,
		updateGoal,
		deleteGoal,
		ApiError,
		type WireGoal,
		type GoalStatus
	} from '$lib/api';
	import { shortWhen } from '$lib/time';

	const app = APPS.find((a) => a.id === 'goals')!;

	// The goals applet: the human-editable standing context the next turn reads.
	// One list, grouped by lifecycle (active / done / abandoned), filtered by
	// ownership scope (all / global / session). The store is the source of truth;
	// every edit posts back to /api/goals and re-reads, so what's shown is what a
	// Construct turn would see.

	type Scope = 'all' | 'global' | 'session';
	const SCOPES: Scope[] = ['all', 'global', 'session'];
	const STATUS_ORDER: GoalStatus[] = ['active', 'done', 'abandoned'];

	let goals = $state<WireGoal[]>([]);
	let total = $state(0);
	let error = $state<string | null>(null);
	let loading = $state(true);

	let scope = $state<Scope>('all');
	// Only meaningful when scope === 'session': which conversation to filter to.
	let sessionFilter = $state('');

	// New-goal composer. A new goal here is global (shared) by default — the most
	// common reason to open this page is to set standing intent every conversation
	// sees; session goals are usually set by the agent itself.
	let draft = $state('');
	let creating = $state(false);

	// Inline content edit: the id being edited and its working buffer.
	let editingId = $state<number | null>(null);
	let editBuffer = $state('');
	let busyId = $state<number | null>(null);

	async function load() {
		loading = true;
		error = null;
		try {
			const opts =
				scope === 'session'
					? { scope, session: sessionFilter || undefined }
					: scope === 'global'
						? { scope }
						: {};
			// scope=session with no session id yet: nothing to show, don't 400.
			if (scope === 'session' && !sessionFilter) {
				goals = [];
				total = 0;
				return;
			}
			const res = await getGoals(opts);
			goals = res.goals;
			total = res.total;
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load goals';
		} finally {
			loading = false;
		}
	}

	// Re-load whenever the scope or session filter changes.
	$effect(() => {
		// reference the deps so the effect tracks them
		void scope;
		void sessionFilter;
		load();
	});

	// Goals split into lifecycle groups, each in oldest-first order (the store's
	// natural to-do order). Derived so it tracks the list live.
	const grouped = $derived.by(() => {
		const out: Record<GoalStatus, WireGoal[]> = { active: [], done: [], abandoned: [] };
		for (const g of goals) out[g.status].push(g);
		return out;
	});

	async function addGoal() {
		const content = draft.trim();
		if (!content || creating) return;
		creating = true;
		error = null;
		try {
			// From this page a goal is global unless we're filtered to a session,
			// in which case it joins that conversation.
			const session = scope === 'session' && sessionFilter ? sessionFilter : undefined;
			await createGoal({ content, session });
			draft = '';
			await load();
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to create goal';
		} finally {
			creating = false;
		}
	}

	function startEdit(g: WireGoal) {
		editingId = g.id;
		editBuffer = g.content;
	}

	function cancelEdit() {
		editingId = null;
		editBuffer = '';
	}

	async function saveEdit(g: WireGoal) {
		const content = editBuffer.trim();
		if (!content || content === g.content) {
			cancelEdit();
			return;
		}
		busyId = g.id;
		error = null;
		try {
			await updateGoal(g.id, { content });
			cancelEdit();
			await load();
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to save goal';
		} finally {
			busyId = null;
		}
	}

	async function setStatus(g: WireGoal, status: GoalStatus) {
		if (g.status === status) return;
		busyId = g.id;
		error = null;
		try {
			await updateGoal(g.id, { status });
			await load();
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to update goal';
		} finally {
			busyId = null;
		}
	}

	async function removeGoal(g: WireGoal) {
		if (!confirm('Delete this goal? Abandon it instead to keep the record of intent.')) return;
		busyId = g.id;
		error = null;
		try {
			await deleteGoal(g.id);
			await load();
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to delete goal';
		} finally {
			busyId = null;
		}
	}

	function onComposerKey(e: KeyboardEvent) {
		// Enter submits; Shift+Enter is a newline (a goal can be a sentence or two).
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			addGoal();
		}
	}
</script>

<AppHeader title={app.title} icon={app.icon}>
	<span class="text-faint text-[10px] lowercase">{loading ? '…' : `${total} total`}</span>
</AppHeader>

<div class="flex min-h-0 flex-1 flex-col">
	<!-- Scope filter + (when scoped to a session) the session id input. -->
	<div class="flex items-center gap-3 border-b border-border px-4 py-2">
		<div class="flex items-center gap-1">
			{#each SCOPES as s (s)}
				<button
					class="rounded px-2 py-0.5 text-[11px] lowercase
						{scope === s ? 'bg-surface text-text' : 'text-muted hover:bg-surface/50'}"
					onclick={() => (scope = s)}>{s}</button
				>
			{/each}
		</div>
		{#if scope === 'session'}
			<input
				class="text-text placeholder:text-faint min-w-0 flex-1 bg-transparent text-[11px] outline-none"
				placeholder="session id…"
				bind:value={sessionFilter}
			/>
		{/if}
	</div>

	<!-- New-goal composer. -->
	<div class="flex items-center gap-2 border-b border-border px-4 py-2">
		<textarea
			class="text-text placeholder:text-faint min-h-[1.5rem] flex-1 resize-none bg-transparent text-xs outline-none"
			rows="1"
			placeholder={scope === 'session' && sessionFilter
				? 'new goal for this conversation…'
				: 'new shared goal…'}
			bind:value={draft}
			onkeydown={onComposerKey}
		></textarea>
		<button
			class="text-glow hover:text-text shrink-0 text-[10px] lowercase disabled:opacity-40"
			disabled={creating || !draft.trim()}
			onclick={addGoal}>{creating ? 'adding…' : '+ add'}</button
		>
	</div>

	{#if error}
		<div class="text-faint border-b border-border/40 px-4 py-1.5 text-[11px]">{error}</div>
	{/if}

	<div class="min-h-0 flex-1 overflow-y-auto">
		{#if !loading && goals.length === 0}
			<div class="text-faint px-4 py-3 text-xs lowercase">
				{scope === 'session' && !sessionFilter ? 'enter a session id to see its goals' : 'no goals here'}
			</div>
		{:else}
			{#each STATUS_ORDER as status (status)}
				{#if grouped[status].length > 0}
					<div class="text-faint border-b border-border/40 px-4 py-1 text-[10px] lowercase tracking-wide">
						{status} · {grouped[status].length}
					</div>
					{#each grouped[status] as g (g.id)}
						<div class="flex items-start gap-3 border-b border-border/40 px-4 py-2 text-xs">
							<span class="text-faint w-8 shrink-0 pt-0.5">#{g.id}</span>
							<div class="min-w-0 flex-1">
								{#if editingId === g.id}
									<!-- svelte-ignore a11y_autofocus -->
									<textarea
										class="text-text w-full resize-none bg-transparent text-xs outline-none"
										rows="2"
										autofocus
										bind:value={editBuffer}
										onkeydown={(e) => {
											if (e.key === 'Enter' && !e.shiftKey) {
												e.preventDefault();
												saveEdit(g);
											} else if (e.key === 'Escape') {
												cancelEdit();
											}
										}}
									></textarea>
									<div class="mt-1 flex gap-3">
										<button
											class="text-glow hover:text-text text-[10px] lowercase"
											onclick={() => saveEdit(g)}>save</button
										>
										<button
											class="text-faint hover:text-text text-[10px] lowercase"
											onclick={cancelEdit}>cancel</button
										>
									</div>
								{:else}
									<button
										class="text-text block w-full text-left {status !== 'active'
											? 'text-muted line-through'
											: ''}"
										title="click to edit"
										onclick={() => startEdit(g)}>{g.content}</button
									>
									<div class="text-faint mt-1 flex flex-wrap items-center gap-2 text-[10px] lowercase">
										{#if g.session}
											<a class="hover:text-text underline" href={`/?session=${encodeURIComponent(g.session)}`}
												>session {g.session.slice(0, 8)}</a
											>
										{:else}
											<span class="text-glow">shared</span>
										{/if}
										<span title={`updated ${shortWhen(g.updated)}`}>{shortWhen(g.updated)}</span>
									</div>
								{/if}
							</div>
							<!-- Status + delete controls. -->
							<div class="flex shrink-0 items-center gap-2 pt-0.5">
								{#each STATUS_ORDER as s (s)}
									{#if s !== g.status}
										<button
											class="text-faint hover:text-text text-[10px] lowercase disabled:opacity-40"
											disabled={busyId === g.id}
											title={`mark ${s}`}
											onclick={() => setStatus(g, s)}>{s === 'active' ? 'reopen' : s}</button
										>
									{/if}
								{/each}
								<button
									class="text-faint hover:text-text text-[10px] lowercase disabled:opacity-40"
									disabled={busyId === g.id}
									title="delete"
									onclick={() => removeGoal(g)}>✕</button
								>
							</div>
						</div>
					{/each}
				{/if}
			{/each}
		{/if}
	</div>
</div>
