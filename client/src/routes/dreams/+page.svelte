<script lang="ts">
	import AppHeader from '$lib/AppHeader.svelte';
	import { APPS } from '$lib/apps';
	import { getDreams, runDreams, ApiError, type WireDream } from '$lib/api';
	import { shortWhen, iso } from '$lib/time';

	const app = APPS.find((a) => a.id === 'dreams')!;

	// Live dreams from GET /api/dreams, newest first. A dream is a disposable
	// persona's choice on a scenario abstracted from the corpus: persona +
	// scenario + choice, the structured record the dream loop wrote to each event.
	// The button drives POST /api/dreams to run more, which the server appends to
	// the log and returns so we prepend them without a re-fetch.
	let dreams = $state<WireDream[]>([]);
	let total = $state(0);
	let error = $state<string | null>(null);
	let loading = $state(true);
	// While a dream batch is in flight: dreaming is several model turns per dream,
	// so it's slow and the button must show it's working (and not double-fire).
	let dreaming = $state(false);

	async function load() {
		loading = true;
		error = null;
		try {
			const res = await getDreams();
			dreams = res.dreams;
			total = res.total;
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to load dreams';
		} finally {
			loading = false;
		}
	}

	// Run one dream now. The server returns the dream(s) it produced (already on
	// the log), so we prepend them — newest first, matching the list order — and
	// bump the total, rather than re-fetching the whole page.
	async function dream() {
		if (dreaming) return;
		dreaming = true;
		error = null;
		try {
			const res = await runDreams({ count: 1 });
			dreams = [...res.dreams, ...dreams];
			total += res.dreams.length;
			// A dream that failed to form (a malformed persona, a blip) isn't fatal
			// to the batch; surface it quietly so an empty result isn't silent.
			if (res.dreams.length === 0 && res.failures.length > 0) {
				error = res.failures[0].error;
			}
		} catch (e) {
			error = e instanceof ApiError ? e.message : 'failed to dream';
		} finally {
			dreaming = false;
		}
	}

	$effect(() => {
		load();
	});

	// The persona's headline: name, then role if it has one. The dreamer often
	// arrives with just a name, so role is folded in only when present.
	function personaLine(d: WireDream): string {
		return d.persona.role ? `${d.persona.name}, ${d.persona.role}` : d.persona.name;
	}

	// Which dreams are expanded. Dreams collapse by default: the scenario and
	// choice are several lines each, so an unfiltered list is a wall of text.
	// Collapsed, each dream is clamped to a couple of lines with a trailing
	// ellipsis ("…") where the text would have wrapped on; clicking the body
	// toggles it open to read in full. Keyed by dream id so the set survives a
	// prepend from + dream.
	let expanded = $state<Set<number>>(new Set());

	function toggle(id: number) {
		const next = new Set(expanded);
		next.has(id) ? next.delete(id) : next.add(id);
		expanded = next;
	}
</script>

<AppHeader title={app.title} icon={app.icon}>
	<button
		class="text-glow hover:text-text text-[10px] lowercase disabled:opacity-50"
		title="dream once: a fresh persona faces a scenario drawn from the corpus"
		onclick={dream}
		disabled={dreaming}>{dreaming ? 'dreaming…' : '+ dream'}</button
	>
	<span class="text-faint text-[10px] lowercase">{loading ? '…' : total}</span>
</AppHeader>

<div class="min-h-0 flex-1 overflow-y-auto">
	{#if error}
		<div class="text-faint px-4 py-3 text-xs">{error}</div>
	{:else if !loading && dreams.length === 0}
		<div class="text-faint px-4 py-3 text-xs lowercase">
			no dreams yet — press <span class="text-glow">+ dream</span> to dream one
		</div>
	{:else}
		{#each dreams as d (d.id)}
			{@const open = expanded.has(d.id)}
			<!-- A collapsible section per dream. The persona/timestamp line is the
			     header; the scenario and choice are the body. Clicking anywhere in
			     the body toggles the section open or shut: collapsed clamps each
			     to a couple of lines with a "…" where the text would have wrapped
			     on, expanded shows it whole. A button so it's keyboard-reachable. -->
			<button
				type="button"
				class="block w-full cursor-pointer border-b border-border/40 px-4 py-3 text-left text-xs hover:bg-surface/30"
				aria-expanded={open}
				onclick={() => toggle(d.id)}
			>
				<div class="flex items-baseline justify-between gap-3">
					<span class="text-muted">{personaLine(d)}</span>
					<span class="text-faint shrink-0 text-[10px] lowercase" title={iso(d.ts)}
						>{shortWhen(d.ts)}</span
					>
				</div>
				<!-- The dilemma the persona faced, then the choice it made. The choice
				     is free prose ("choose, and say why"), not a verdict, so it reads
				     as the dreamer's reasoning rather than a pass/fail. When collapsed,
				     line-clamp truncates each with a trailing ellipsis. -->
				<div class="text-text mt-1 {open ? '' : 'line-clamp-2'}">{d.scenario}</div>
				<div
					class="text-muted border-glow/30 mt-2 border-l-2 pl-3 whitespace-pre-wrap {open
						? ''
						: 'line-clamp-2'}"
				>
					{d.choice}
				</div>
			</button>
		{/each}
	{/if}
</div>
