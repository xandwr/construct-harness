<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import {
		getMemories,
		getMemory,
		updateMemory,
		deleteMemory,
		ApiError,
		type WireMemory,
		type WireEvent
	} from '$lib/api';
	import { shortWhen } from '$lib/time';

	const app = APPS.find((a) => a.id === 'memories')!;

	// The curated memory store as a list + detail pane: the list on the left, an
	// inspect/edit pane on the right. This is memory *provenance*, so the detail
	// shows what the agent's flat table couldn't — earned strength, when the memory
	// last surfaced, whether it's embedded, and the conversation it was curated
	// from (with a jump). The store is the source of truth; this curates it.

	let rows = $state<WireMemory[]>([]);
	let total = $state(0);
	let error = $state<string | null>(null);
	let loading = $state(true);
	let query = $state('');

	// The open memory (detail) and its editable buffers.
	let selectedId = $state<number | null>(null);
	let open = $state<WireMemory | null>(null);
	let sourceEvent = $state<WireEvent | null>(null);
	let draftContent = $state('');
	let draftTags = $state('');
	let draftImportance = $state('');
	let dirty = $state(false);
	let saving = $state(false);
	let paneError = $state<string | null>(null);

	async function load() {
		loading = true;
		error = null;
		try {
			const res = await getMemories({ q: query || undefined });
			rows = res.memories;
			total = res.total;
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load memories';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		load();
	});

	async function select(id: number) {
		if (dirty && !confirm('Discard unsaved changes?')) return;
		selectedId = id;
		paneError = null;
		try {
			const res = await getMemory(id);
			open = res.memory;
			sourceEvent = res.sourceEvent;
			draftContent = res.memory.content;
			draftTags = res.memory.tags.join(' ');
			draftImportance = res.memory.importance === null ? '' : String(res.memory.importance);
			dirty = false;
		} catch (e) {
			paneError = e instanceof ApiError ? e.message : 'failed to open the memory';
			open = null;
		}
	}

	function markDirty() {
		dirty = true;
	}

	async function save() {
		if (!open) return;
		saving = true;
		paneError = null;
		try {
			// Importance: blank clears it (null), else parse a number. A non-numeric
			// non-blank value is rejected before we hit the wire.
			let importance: number | null | undefined;
			const trimmed = draftImportance.trim();
			if (trimmed === '') importance = null;
			else {
				const n = Number(trimmed);
				if (!Number.isFinite(n)) {
					paneError = 'importance must be a number, or blank to clear it';
					saving = false;
					return;
				}
				importance = n;
			}
			const tags = draftTags
				.split(/[\s,]+/)
				.map((t) => t.trim())
				.filter(Boolean);
			const res = await updateMemory(open.id, { content: draftContent, tags, importance });
			open = res.memory;
			dirty = false;
			await load();
		} catch (e) {
			paneError = e instanceof ApiError ? e.message : 'failed to save';
		} finally {
			saving = false;
		}
	}

	async function forget() {
		if (!open) return;
		if (!confirm('Forget this memory? This removes it from the store.')) return;
		try {
			await deleteMemory(open.id);
			open = null;
			selectedId = null;
			sourceEvent = null;
			dirty = false;
			await load();
		} catch (e) {
			paneError = e instanceof ApiError ? e.message : 'failed to forget';
		}
	}

	// The deep-link to the source conversation, scrolled to the originating event.
	// Mirrors the chat page's ?session= + #event-<id> contract.
	function sourceLink(m: WireMemory): string | null {
		if (!m.provenance?.session) return null;
		return `/?session=${encodeURIComponent(m.provenance.session)}#event-${m.provenance.eventId}`;
	}

	function onKeydown(e: KeyboardEvent) {
		if ((e.ctrlKey || e.metaKey) && e.key === 's') {
			e.preventDefault();
			if (dirty && open) save();
		}
	}
</script>

<svelte:window onkeydown={onKeydown} />

<AppHeader title={app.title} icon={app.icon}>
	<span class="text-faint text-[10px] lowercase">{loading ? '…' : total}</span>
</AppHeader>

<div class="flex min-h-0 flex-1">
	<!-- List -->
	<div class="flex w-80 shrink-0 flex-col border-r border-border">
		<div class="border-b border-border px-2 py-1.5">
			<input
				class="text-text placeholder:text-faint w-full bg-transparent text-[11px] outline-none"
				placeholder="search memories…"
				bind:value={query}
			/>
		</div>
		<div class="min-h-0 flex-1 overflow-y-auto">
			{#if error}
				<div class="text-faint px-3 py-3 text-xs">{error}</div>
			{:else if !loading && rows.length === 0}
				<div class="text-faint px-3 py-3 text-xs lowercase">no memories yet</div>
			{:else}
				{#each rows as m (m.id)}
					<button
						class="block w-full border-b border-border/40 px-3 py-2 text-left
							{selectedId === m.id ? 'bg-surface' : 'hover:bg-surface/50'}"
						onclick={() => select(m.id)}
					>
						<div class="text-text line-clamp-2 text-xs">{m.content}</div>
						<div class="text-faint mt-1 flex items-center gap-2 text-[10px] lowercase">
							<span class="text-muted">#{m.id}</span>
							{#if m.tags.length}<span>{m.tags.join(' ')}</span>{/if}
							<span title="effective strength">str {m.strength.toFixed(2)}</span>
							{#if m.hasEmbedding}<span title="has an embedding">vec</span>{/if}
						</div>
					</button>
				{/each}
			{/if}
		</div>
	</div>

	<!-- Detail / edit pane -->
	<div class="flex min-w-0 flex-1 flex-col">
		{#if !open}
			<div class="text-faint flex flex-1 items-center justify-center text-xs lowercase">
				select a memory to inspect
			</div>
		{:else}
			<div class="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
				<span class="text-faint text-[10px] lowercase">memory #{open.id}</span>
				<div class="flex shrink-0 items-center gap-3">
					{#if dirty}<span class="text-glow text-[10px] lowercase">unsaved</span>{/if}
					<button
						class="text-glow hover:text-text text-[10px] lowercase disabled:opacity-40"
						disabled={!dirty || saving}
						onclick={save}>{saving ? 'saving…' : 'save'}</button
					>
					<button class="text-faint hover:text-text text-[10px] lowercase" onclick={forget}
						>forget</button
					>
				</div>
			</div>
			{#if paneError}
				<div class="text-faint border-b border-border/40 px-3 py-1.5 text-[11px]">{paneError}</div>
			{/if}

			<div class="min-h-0 flex-1 overflow-y-auto px-3 py-3">
				<!-- Content -->
				<label class="text-faint mb-1 block text-[10px] lowercase">
					content
					<textarea
						class="text-text mt-1 min-h-20 w-full resize-y bg-transparent text-xs leading-relaxed outline-none"
						spellcheck="false"
						bind:value={draftContent}
						oninput={markDirty}
					></textarea>
				</label>

				<!-- Tags + importance -->
				<div class="mb-3 mt-3 flex gap-4">
					<label class="text-faint min-w-0 flex-1 text-[10px] lowercase">
						tags (space-separated)
						<input
							class="text-text mt-1 w-full bg-transparent text-xs outline-none"
							bind:value={draftTags}
							oninput={markDirty}
						/>
					</label>
					<label class="text-faint w-24 shrink-0 text-[10px] lowercase">
						importance
						<input
							class="text-text mt-1 w-full bg-transparent text-xs outline-none"
							placeholder="—"
							bind:value={draftImportance}
							oninput={markDirty}
						/>
					</label>
				</div>

				<!-- Provenance / signals -->
				<div class="border-t border-border/40 pt-3 text-[11px]">
					<div class="text-faint mb-2 text-[10px] lowercase">provenance</div>
					<dl class="grid grid-cols-[7rem_1fr] gap-y-1">
						<dt class="text-faint">strength</dt>
						<dd class="text-muted">{open.strength.toFixed(3)}</dd>
						<dt class="text-faint">last surfaced</dt>
						<dd class="text-muted">{open.lastSurfaced === null ? 'never' : shortWhen(open.lastSurfaced)}</dd>
						<dt class="text-faint">embedding</dt>
						<dd class="text-muted">{open.hasEmbedding ? 'present' : 'none (lexical only)'}</dd>
						<dt class="text-faint">created</dt>
						<dd class="text-muted">{shortWhen(open.created)}</dd>
						<dt class="text-faint">updated</dt>
						<dd class="text-muted">{shortWhen(open.updated)}</dd>
						<dt class="text-faint">source</dt>
						<dd class="text-muted">
							{#if open.provenance}
								{#if sourceLink(open)}
									<a class="hover:text-text underline" href={sourceLink(open)}
										>event #{open.provenance.eventId} · session {open.provenance.session?.slice(0, 8)}</a
									>
								{:else}
									event #{open.provenance.eventId}
								{/if}
							{:else}
								—
							{/if}
						</dd>
					</dl>
					{#if sourceEvent}
						<div class="text-faint mt-3 mb-1 text-[10px] lowercase">curated from</div>
						<pre
							class="text-muted whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">{sourceEvent.content}</pre>
					{/if}
				</div>
			</div>
		{/if}
	</div>
</div>
