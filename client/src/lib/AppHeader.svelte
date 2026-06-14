<script lang="ts">
	// The applet's title bar: the app's own icon, its name, and an optional status
	// slot. The icon is the same glyph the sidebar uses (passed as a name), so the
	// active applet reads the same in the spine and in its own header. No
	// description, no ornament. Render only what is necessary.
	import Icon from '$lib/Icon.svelte';

	let {
		title,
		icon,
		children
	}: {
		title: string;
		icon?: string;
		children?: import('svelte').Snippet;
	} = $props();
</script>

<header class="flex items-center justify-between gap-4 border-b border-border px-4 py-2.5">
	<div class="flex items-center gap-2">
		{#if icon}
			<Icon name={icon} class="text-glow size-3.5 shrink-0" />
		{/if}
		<h1 class="text-text text-xs lowercase tracking-wide">{title}</h1>
	</div>
	{#if children}
		<div class="flex shrink-0 items-center gap-2">
			{@render children()}
		</div>
	{/if}
</header>
