import { ChatService } from '$lib/services/chat.service';
import { MessageRole } from '$lib/enums';
import { afterEach, describe, expect, it, vi } from 'vitest';

describe('ChatService prefix cache requests', () => {
	afterEach(() => {
		vi.unstubAllGlobals();
		vi.restoreAllMocks();
	});

	it('builds precache requests through the shared chat completion body', async () => {
		const fetchMock = vi.fn().mockResolvedValue(
			new Response(
				JSON.stringify({
					attention_blocks: 1,
					boundary: 2048,
					bytes_published: 123,
					durable: true,
					model: 'qwen',
					reason: '',
					recurrent_sidecars: 1,
					tokens_evaluated: 2049
				}),
				{ status: 200 }
			)
		);

		vi.stubGlobal('fetch', fetchMock);

		await ChatService.precachePrefix(
			[
				{
					content: 'hello',
					reasoning_content: 'private thinking',
					role: MessageRole.USER
				}
			],
			{
				custom: '{"chat_template_kwargs":{"enable_thinking":false}}',
				excludeReasoningFromContext: true,
				model: 'qwen',
				stream: true,
				systemMessage: 'system prompt',
				temperature: 0.2
			}
		);

		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(fetchMock.mock.calls[0][0]).toBe('./cache/prefix');

		const init = fetchMock.mock.calls[0][1] as RequestInit;
		const body = JSON.parse(String(init.body));

		expect(init.method).toBe('POST');
		expect(body).toMatchObject({
			chat_template_kwargs: { enable_thinking: false },
			model: 'qwen',
			reasoning_control: true,
			reasoning_format: 'auto',
			stream: false,
			temperature: 0.2
		});
		expect(body.messages).toEqual([
			{ content: 'system prompt', role: MessageRole.SYSTEM },
			{
				content: 'hello',
				role: MessageRole.USER
			}
		]);
	});

	it('throws the stable server error reason when precache fails', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(
				new Response(JSON.stringify({ error: { message: 'persistent prefix cache is disabled' } }), {
					status: 400,
					statusText: 'Bad Request'
				})
			)
		);

		await expect(
			ChatService.precachePrefix([{ content: 'hello', role: MessageRole.USER }])
		).rejects.toThrow('persistent prefix cache is disabled');
	});
});
