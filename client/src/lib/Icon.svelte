<script lang="ts" module>
	// All icon SVGs, imported as raw strings at build time. `eager` so they're
	// inlined into the bundle (no per-icon network fetch) and resolved at module
	// load, not lazily. The glob key is the file path; we re-key it by basename
	// (the icon `name`) below.
	const files = import.meta.glob('./icons/*.svg', {
		query: '?raw',
		import: 'default',
		eager: true
	}) as Record<string, string>;

	/**
	 * Normalize one downloaded SVG so it renders as a sized, color-inheriting
	 * glyph regardless of how the source was authored. The icon set is mixed:
	 * some are stroked outlines, some are solid fills, on differing viewBoxes. We
	 * deliberately do NOT rewrite the artwork; we only:
	 *   - drop fixed width/height so the CSS box (set on the wrapping <svg>'s
	 *     parent) controls size,
	 *   - rewrite any concrete stroke/fill color to `currentColor` so the icon
	 *     takes the surrounding text color (the breathe/glow/muted states),
	 *   - strip the XML prolog and comments so {@html} injects clean markup.
	 * Each file keeps its own viewBox and its own fill-vs-stroke nature.
	 */
	function normalize(raw: string): string {
		return (
			raw
				// XML declaration and generator comments: not needed inline.
				.replace(/<\?xml[^>]*\?>/gi, '')
				.replace(/<!--[\s\S]*?-->/g, '')
				// Fixed pixel dimensions: let the layout size it via the parent.
				.replace(/\s(width|height)="[^"]*"/gi, '')
				// Any concrete color -> currentColor, so it inherits text color.
				// Covers fill="#000000", stroke="#000", fill="black", etc. Leaves
				// fill="none" alone (that's structural, not a color choice).
				.replace(/(fill|stroke)="(?!none")[^"]*"/gi, '$1="currentColor"')
				.trim()
		);
	}

	// name (basename, e.g. "event-log") -> normalized inline svg string.
	const ICONS: Record<string, string> = {};
	for (const [path, raw] of Object.entries(files)) {
		const name = path.split('/').pop()!.replace('.svg', '');
		ICONS[name] = normalize(raw);
	}

	/** The icon names available, for callers that want to validate. */
	export const ICON_NAMES = Object.keys(ICONS);
</script>

<script lang="ts">
	// Renders a named icon inline. Size and color come from the surrounding
	// context: set a size utility (e.g. `size-4`) and a text color on this
	// component's class, and the glyph follows. `title` adds an accessible label;
	// without one the icon is decorative (aria-hidden).
	let {
		name,
		class: klass = '',
		title
	}: { name: string; class?: string; title?: string } = $props();

	const markup = $derived(ICONS[name]);
</script>

{#if markup}
	<span
		class="icon inline-flex {klass}"
		role={title ? 'img' : undefined}
		aria-label={title}
		aria-hidden={title ? undefined : 'true'}
	>
		<!-- normalized at build time from a trusted local asset; no user input -->
		{@html markup}
	</span>
{/if}

<style>
	/* The injected <svg> fills the wrapper, which the parent sizes. currentColor
	   (set by normalize) means text color drives both stroke and fill. */
	.icon :global(svg) {
		width: 100%;
		height: 100%;
		display: block;
	}
</style>
