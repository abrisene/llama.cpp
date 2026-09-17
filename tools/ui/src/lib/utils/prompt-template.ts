import nunjucks from 'nunjucks';

const TEMPLATE_MARKER_REGEX = /\{\{|\{%/;
const EXPRESSION_ONLY_REGEX = /\{\{/;

const env = new nunjucks.Environment(null, { autoescape: false });

env.addGlobal('pick', (...items: unknown[]): unknown => {
	if (items.length === 0) return '';

	return items[Math.floor(Math.random() * items.length)];
});

env.addFilter('random', (items: unknown[], count?: number): unknown => {
	if (!Array.isArray(items) || items.length === 0) return '';

	if (count !== undefined && count > 1) {
		const shuffled = [...items].sort(() => Math.random() - 0.5);

		return shuffled.slice(0, Math.min(count, items.length));
	}

	return items[Math.floor(Math.random() * items.length)];
});

// Fenced code blocks (``` or ~~~) — always protected so templates
// inside code examples are never resolved.
const FENCED_CODE_REGEX = /(?:^|\n)(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;

// Full protection: fenced code blocks, inline code (`...`), and
// blockquotes (lines starting with >). Used for user/system messages
// where the author deliberately placed template syntax in those regions.
const FULL_PROTECTED_REGEX =
	/(?:^|\n)(```[\s\S]*?```|~~~[\s\S]*?~~~)|`[^`\n]+`|(?:^|\n)((?:>.*(?:\n|$))+)/g;

// Matches {% ... %} control blocks so they can be shielded when only
// expression blocks ({{ ... }}) should be resolved.
const CONTROL_BLOCK_REGEX = /\{%[\s\S]*?%\}/g;

function shieldRegions(source: string, regex: RegExp): { masked: string; slots: string[] } {
	const slots: string[] = [];

	const masked = source.replace(regex, (match) => {
		const idx = slots.length;
		slots.push(match);

		return `\x00PROT_${idx}\x00`;
	});

	return { masked, slots };
}

function shieldControlBlocks(source: string, slots: string[]): string {
	return source.replace(CONTROL_BLOCK_REGEX, (match) => {
		const idx = slots.length;
		slots.push(match);

		return `\x00PROT_${idx}\x00`;
	});
}

function restoreProtectedRegions(rendered: string, slots: string[]): string {
	return rendered.replace(/\x00PROT_(\d+)\x00/g, (_, idx) => slots[Number(idx)]);
}

export interface ResolveOptions {
	expressionsOnly?: boolean;
}

/**
 * Resolves a Jinja2-style prompt template (pools via `pick(...)` / `| random`,
 * plus whatever Nunjucks control flow the author writes) into literal text.
 *
 * For user/system messages (default): template syntax inside fenced code blocks,
 * inline code, and blockquotes is preserved verbatim.
 *
 * For assistant responses (`expressionsOnly`): only fenced code blocks are
 * protected, and only `{{ ... }}` expression blocks are resolved — `{% ... %}`
 * control blocks are preserved since the model may be writing Jinja for a prompt.
 */
export function resolvePromptTemplate(source: string, opts?: ResolveOptions): string {
	const markerRegex = opts?.expressionsOnly ? EXPRESSION_ONLY_REGEX : TEMPLATE_MARKER_REGEX;

	if (!markerRegex.test(source)) return source;

	try {
		const protectionRegex = opts?.expressionsOnly ? FENCED_CODE_REGEX : FULL_PROTECTED_REGEX;
		const { masked, slots } = shieldRegions(source, protectionRegex);
		let toRender = masked;

		if (opts?.expressionsOnly) {
			toRender = shieldControlBlocks(toRender, slots);
		}

		if (!markerRegex.test(toRender)) return source;

		return restoreProtectedRegions(env.renderString(toRender, {}), slots);
	} catch (error) {
		console.warn('[prompt-template] Failed to resolve template, sending raw text:', error);

		return source;
	}
}

/**
 * Checks whether a (possibly incomplete) streamed string contains a complete
 * `{{ ... }}` expression that can be resolved. Returns the index of the first
 * `{{` whose `}}` has been received, or -1 if none is ready yet.
 *
 * Only fenced code blocks are protected (not inline backticks), since this
 * is used for assistant response content.
 */
export function findCompleteExpression(source: string): number {
	const { masked } = shieldRegions(source, FENCED_CODE_REGEX);
	const re = /\{\{[\s\S]*?\}\}/g;
	const match = re.exec(masked);

	return match ? match.index : -1;
}
