<script lang="ts">
	// The applet's title bar: just the name and an optional status slot. No
	// description, no ornament. Render only what is necessary.
	let {
		title,
		children
	}: { title: string; children?: import('svelte').Snippet } = $props();
</script>

<header class="flex items-center justify-between gap-4 border-b border-border px-4 py-2.5">
	<h1 class="text-text text-xs lowercase tracking-wide">{title}</h1>
	{#if children}
		<div class="flex shrink-0 items-center gap-2">
			{@render children()}
		</div>
	{/if}
</header>
