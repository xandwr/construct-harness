<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import { page } from '$app/state';
	import { APPS, activeApp } from '$lib/apps';
	import Icon from '$lib/Icon.svelte';
	import UserStatus from '$lib/UserStatus.svelte';
	import '../style.css';

	let { children } = $props();

	const current = $derived(activeApp(page.url.pathname));
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>{current ? `construct · ${current.title}` : 'construct'}</title>
</svelte:head>

<div class="flex h-screen w-screen overflow-hidden">
	<nav class="flex w-48 shrink-0 flex-col border-r border-border bg-base" aria-label="applets">
		<div class="flex items-center gap-2 border-b border-border px-3 py-3">
			<img src={favicon} alt="Logo" class="w-5 h-5">
			<span class="text-text text-xs lowercase tracking-wide">construct</span>
		</div>

		<ul class="flex flex-col">
			{#each APPS as app (app.id)}
				{@const isActive = current?.id === app.id}
				<li>
					<a
						href={app.href}
						aria-current={isActive ? 'page' : undefined}
						class="group flex items-center gap-2.5 border-b border-border/40 px-3 py-2
							{isActive ? 'bg-surface text-text' : 'text-muted hover:bg-surface/50 hover:text-text'}"
					>
						<Icon
							name={app.icon}
							class="size-3.5 shrink-0 {isActive
								? 'text-glow'
								: 'text-faint group-hover:text-muted'}"
						/>
						<span class="text-xs lowercase">{app.title}</span>
					</a>
				</li>
			{/each}
		</ul>

		<!-- The human's presence pins to the bottom of the spine, where Discord puts
		     yours: the spacer pushes it down, the rule sets it off from the apps. -->
		<div class="mt-auto border-t border-border">
			<UserStatus />
		</div>
	</nav>

	<!-- The active applet. -->
	<main class="flex min-w-0 flex-1 flex-col bg-base">
		{@render children()}
	</main>
</div>
