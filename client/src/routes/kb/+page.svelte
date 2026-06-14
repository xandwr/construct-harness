<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import {
		getNotes,
		getNote,
		createNote,
		updateNote,
		deleteNote,
		ApiError,
		type WireNoteSummary,
		type WireNote
	} from '$lib/api';
	import { shortWhen } from '$lib/time';

	const app = APPS.find((a) => a.id === 'kb')!;

	// The knowledge base, three panes: a folder tree (derived from note paths) on
	// the left, the note list for the selected folder in the middle, and a
	// view/edit pane on the right that posts back to the write API. The store is
	// the source of truth; this is an instrument over it, mirroring the other
	// applets' read-table shape with editing on top.

	let notes = $state<WireNoteSummary[]>([]);
	let total = $state(0);
	let error = $state<string | null>(null);
	let loading = $state(true);

	// Folder filter (a path prefix like "ops/"; '' is the root, showing all).
	let folder = $state('');
	let query = $state('');

	// The open note (detail shape, with body) and its editable buffers.
	let selectedUuid = $state<string | null>(null);
	let open = $state<WireNote | null>(null);
	let draftTitle = $state('');
	let draftBody = $state('');
	let dirty = $state(false);
	let saving = $state(false);
	let paneError = $state<string | null>(null);

	async function load() {
		loading = true;
		error = null;
		try {
			const res = await getNotes({ q: query || undefined });
			notes = res.notes;
			total = res.total;
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load the knowledge base';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		load();
	});

	// The folder tree: every distinct directory prefix across all note paths,
	// sorted, plus the implicit root. Derived, so it tracks the note list live.
	const folders = $derived.by(() => {
		const set = new Set<string>();
		for (const n of notes) {
			const parts = n.path.split('/');
			parts.pop(); // drop the filename
			let acc = '';
			for (const part of parts) {
				acc += part + '/';
				set.add(acc);
			}
		}
		return [...set].sort();
	});

	// Notes within the selected folder (or all, at the root), matching the search.
	const visible = $derived(
		notes.filter((n) => (folder === '' ? true : n.path.startsWith(folder)))
	);

	async function selectNote(uuid: string) {
		if (dirty && !confirm('Discard unsaved changes?')) return;
		selectedUuid = uuid;
		paneError = null;
		try {
			const res = await getNote(uuid);
			open = res.note;
			draftTitle = res.note.title;
			draftBody = res.note.content;
			dirty = false;
		} catch (e) {
			paneError = e instanceof ApiError ? e.message : 'failed to open the note';
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
			const res = await updateNote(open.uuid, { title: draftTitle, content: draftBody });
			open = res.note;
			dirty = false;
			await load(); // refresh the list (title/updated may have changed)
		} catch (e) {
			paneError = e instanceof ApiError ? e.message : 'failed to save';
		} finally {
			saving = false;
		}
	}

	async function newNote() {
		const title = prompt('New note title:')?.trim();
		if (!title) return;
		paneError = null;
		try {
			// A new note lands in the currently-selected folder.
			const path = folder ? `${folder}${slug(title)}.md` : undefined;
			const res = await createNote({ title, content: '', path });
			await load();
			await selectNote(res.note.uuid);
		} catch (e) {
			paneError = e instanceof ApiError ? e.message : 'failed to create the note';
		}
	}

	async function removeNote() {
		if (!open) return;
		if (!confirm(`Delete "${open.title}"? This removes the file too.`)) return;
		try {
			await deleteNote(open.uuid);
			open = null;
			selectedUuid = null;
			dirty = false;
			await load();
		} catch (e) {
			paneError = e instanceof ApiError ? e.message : 'failed to delete';
		}
	}

	function slug(title: string): string {
		return (
			title
				.toLowerCase()
				.replace(/[^a-z0-9]+/g, '-')
				.replace(/^-+|-+$/g, '')
				.slice(0, 80) || 'note'
		);
	}

	// Ctrl/Cmd+S saves the open note.
	function onKeydown(e: KeyboardEvent) {
		if ((e.ctrlKey || e.metaKey) && e.key === 's') {
			e.preventDefault();
			if (dirty && open) save();
		}
	}
</script>

<svelte:window onkeydown={onKeydown} />

<AppHeader title={app.title} icon={app.icon}>
	<button
		class="text-glow hover:text-text text-[10px] lowercase"
		title="new note"
		onclick={newNote}>+ new</button
	>
	<span class="text-faint text-[10px] lowercase">{loading ? '…' : total}</span>
</AppHeader>

<div class="flex min-h-0 flex-1">
	<!-- Folder tree -->
	<aside class="flex w-44 shrink-0 flex-col overflow-y-auto border-r border-border">
		<button
			class="border-b border-border/40 px-3 py-1.5 text-left text-[11px] lowercase
				{folder === '' ? 'bg-surface text-text' : 'text-muted hover:bg-surface/50'}"
			onclick={() => (folder = '')}>all notes</button
		>
		{#each folders as f (f)}
			<button
				class="border-b border-border/40 px-3 py-1.5 text-left text-[11px]
					{folder === f ? 'bg-surface text-text' : 'text-muted hover:bg-surface/50'}"
				style="padding-left: {0.75 + (f.split('/').length - 2) * 0.6}rem"
				onclick={() => (folder = f)}
				title={f}
			>
				{f.split('/').filter(Boolean).pop()}/
			</button>
		{/each}
	</aside>

	<!-- Note list -->
	<div class="flex w-72 shrink-0 flex-col border-r border-border">
		<div class="border-b border-border px-2 py-1.5">
			<input
				class="text-text placeholder:text-faint w-full bg-transparent text-[11px] outline-none"
				placeholder="search…"
				bind:value={query}
			/>
		</div>
		<div class="min-h-0 flex-1 overflow-y-auto">
			{#if error}
				<div class="text-faint px-3 py-3 text-xs">{error}</div>
			{:else if !loading && visible.length === 0}
				<div class="text-faint px-3 py-3 text-xs lowercase">no notes here</div>
			{:else}
				{#each visible as n (n.uuid)}
					<button
						class="block w-full border-b border-border/40 px-3 py-2 text-left
							{selectedUuid === n.uuid ? 'bg-surface' : 'hover:bg-surface/50'}"
						onclick={() => selectNote(n.uuid)}
					>
						<div class="text-text truncate text-xs">{n.title}</div>
						<div class="text-faint mt-0.5 truncate text-[10px]">{n.path}</div>
						<div class="text-faint mt-0.5 text-[10px] lowercase">{shortWhen(n.updated)}</div>
					</button>
				{/each}
			{/if}
		</div>
	</div>

	<!-- Edit pane -->
	<div class="flex min-w-0 flex-1 flex-col">
		{#if !open}
			<div class="text-faint flex flex-1 items-center justify-center text-xs lowercase">
				select a note, or create one
			</div>
		{:else}
			<div class="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
				<input
					class="text-text min-w-0 flex-1 bg-transparent text-xs outline-none"
					bind:value={draftTitle}
					oninput={markDirty}
				/>
				<div class="flex shrink-0 items-center gap-3">
					{#if dirty}<span class="text-glow text-[10px] lowercase">unsaved</span>{/if}
					<button
						class="text-glow hover:text-text text-[10px] lowercase disabled:opacity-40"
						disabled={!dirty || saving}
						onclick={save}>{saving ? 'saving…' : 'save'}</button
					>
					<button
						class="text-faint hover:text-text text-[10px] lowercase"
						onclick={removeNote}>delete</button
					>
				</div>
			</div>
			{#if paneError}
				<div class="text-faint border-b border-border/40 px-3 py-1.5 text-[11px]">{paneError}</div>
			{/if}
			<div class="text-faint flex gap-3 border-b border-border/40 px-3 py-1 text-[10px]">
				<span title="relative path in the KB folder">{open.path}</span>
				{#if open.links.length}<span>· {open.links.length} link{open.links.length > 1 ? 's' : ''}</span>{/if}
			</div>
			<textarea
				class="text-text min-h-0 flex-1 resize-none bg-transparent px-3 py-2 text-[12px] leading-relaxed outline-none"
				spellcheck="false"
				bind:value={draftBody}
				oninput={markDirty}
			></textarea>
			{#if open.links.length}
				<div class="border-t border-border px-3 py-2 text-[11px]">
					<div class="text-faint mb-1 text-[10px] lowercase">related</div>
					{#each open.links as l (l.id)}
						<div class="text-muted">
							{l.kind ?? 'links'} →
							{#if l.toNote !== null}note #{l.toNote}{:else}memory #{l.toMemory}{/if}
						</div>
					{/each}
				</div>
			{/if}
		{/if}
	</div>
</div>
