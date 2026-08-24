import ChatFormActionAddDropdownHarness from './components/ChatFormActionAddDropdownHarness.svelte';
import { describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

async function openMenu(screen: Awaited<ReturnType<typeof render>>) {
	await screen.getByRole('button', { name: /add files/i }).click();
}

function precacheAction(screen: Awaited<ReturnType<typeof render>>) {
	return screen.getByText('Precache prefix');
}

describe('ChatForm precache prefix action', () => {
	it('omits the action when persistent prefix cache is not enabled', async () => {
		const onPrecachePrefix = vi.fn();
		const screen = await render(ChatFormActionAddDropdownHarness, {
			onPrecachePrefix,
			showPrecachePrefix: false
		});

		await openMenu(screen);

		await expect.element(precacheAction(screen)).not.toBeInTheDocument();
	});

	it('calls the low-frequency overflow action when enabled', async () => {
		const onPrecachePrefix = vi.fn().mockResolvedValue(undefined);
		const screen = await render(ChatFormActionAddDropdownHarness, {
			onPrecachePrefix,
			showPrecachePrefix: true
		});

		await openMenu(screen);
		await precacheAction(screen).click();

		expect(onPrecachePrefix).toHaveBeenCalledTimes(1);
		await expect.element(screen.getByText('Prefix cached')).toBeVisible();
	});

	it('does not invoke the action while disabled during generation', async () => {
		const onPrecachePrefix = vi.fn().mockResolvedValue(undefined);
		const screen = await render(ChatFormActionAddDropdownHarness, {
			onPrecachePrefix,
			precachePrefixDisabled: true,
			showPrecachePrefix: true
		});

		await openMenu(screen);
		await precacheAction(screen).click();

		expect(onPrecachePrefix).not.toHaveBeenCalled();
	});
});
