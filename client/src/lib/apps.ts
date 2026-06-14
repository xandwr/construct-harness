/**
 * The app registry: the single source of truth for which applets exist in the
 * client and how they appear in the sidebar.
 *
 * The client is a small SWA over the harness, organized as a set of independent
 * applets. Each applet is a SvelteKit route folder that renders in the layout's
 * body; this list is what the sidebar maps over to link them. Adding an applet
 * is two steps: a route folder under `src/routes/`, and one entry here.
 *
 * Names are plain and immediately legible. The interface does not perform; it is
 * an instrument over the harness, and reads like one.
 *
 * `icon` names a file in `src/lib/icons/` (without extension); {@link Icon}
 * resolves it. Adding an applet's icon is dropping a `.svg` in that folder.
 */

export interface AppDef {
	/** Stable id, also the active-route key. Matches the route segment. */
	id: string;
	/** Sidebar label. Plain, lowercase, legible. */
	title: string;
	/** Route to navigate to. Usually `/${id}`; chat owns the index. */
	href: string;
	/** Icon name: a file basename in `src/lib/icons/`, resolved by Icon.svelte. */
	icon: string;
}

/** The applets, in sidebar order. chat owns the index route (`/`). */
export const APPS: AppDef[] = [
	{ id: 'chat', title: 'chat', href: '/', icon: 'chat' },
	{ id: 'conversations', title: 'conversations', href: '/conversations', icon: 'conversations' },
	{ id: 'memories', title: 'memories', href: '/memories', icon: 'memories' },
	{ id: 'log', title: 'event log', href: '/log', icon: 'event-log' },
	{ id: 'dreams', title: 'dreams', href: '/dreams', icon: 'dreams' },
	{ id: 'settings', title: 'settings', href: '/settings', icon: 'settings' }
];

/** Find the applet whose route is currently active, given a pathname. The index
 *  (`/`) maps to chat; everything else matches on the leading path segment so a
 *  nested route still highlights its applet. */
export function activeApp(pathname: string): AppDef | undefined {
	if (pathname === '/') return APPS.find((a) => a.id === 'chat');
	const seg = pathname.split('/').filter(Boolean)[0];
	return APPS.find((a) => a.id === seg);
}
