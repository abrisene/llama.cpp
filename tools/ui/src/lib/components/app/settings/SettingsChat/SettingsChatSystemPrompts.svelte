<script lang="ts">
	import { Copy, Plus, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Select from '$lib/components/ui/select';
	import { Textarea } from '$lib/components/ui/textarea';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { DatabaseService } from '$lib/services/database.service';
	import { debounce } from '$lib/utils';
	import { onMount } from 'svelte';

	interface Props {
		activeId: string;
		onSelectActive: (id: string) => void;
	}

	let { activeId, onSelectActive }: Props = $props();

	const CONTENT_DEBOUNCE_MS = 400;

	let profiles = $state<DatabaseSystemPrompt[]>([]);
	let loading = $state(true);

	let selectedProfile = $derived(profiles.find((p) => p.id === activeId) ?? profiles[0] ?? null);

	async function refresh(): Promise<void> {
		profiles = await DatabaseService.listSystemPrompts();
	}

	onMount(async () => {
		await refresh();

		if (profiles.length === 0) {
			const created = await DatabaseService.createSystemPrompt('Default', '');

			profiles = [created];
		}

		if (!profiles.some((p) => p.id === activeId)) {
			onSelectActive(profiles[0].id);
		}

		loading = false;
	});

	const commitContent = debounce(async (id: string, content: string) => {
		await DatabaseService.updateSystemPrompt(id, { content });
		await refresh();
	}, CONTENT_DEBOUNCE_MS);

	function handleContentInput(value: string) {
		if (!selectedProfile) return;

		const id = selectedProfile.id;

		profiles = profiles.map((p) => (p.id === id ? { ...p, content: value } : p));
		commitContent(id, value);
	}

	async function handleRename(value: string) {
		if (!selectedProfile) return;

		const name = value.trim() || 'Untitled';

		await DatabaseService.updateSystemPrompt(selectedProfile.id, { name });
		await refresh();
	}

	async function handleNew() {
		const created = await DatabaseService.createSystemPrompt('New profile', '');

		await refresh();
		onSelectActive(created.id);
	}

	async function handleDuplicate() {
		if (!selectedProfile) return;

		const created = await DatabaseService.createSystemPrompt(
			`${selectedProfile.name} (copy)`,
			selectedProfile.content
		);

		await refresh();
		onSelectActive(created.id);
	}

	async function handleDelete() {
		if (!selectedProfile || profiles.length <= 1) return;

		const target = selectedProfile;
		const confirmed = window.confirm(`Delete system prompt profile "${target.name}"?`);

		if (!confirmed) return;

		const remaining = profiles.filter((p) => p.id !== target.id);

		await DatabaseService.deleteSystemPrompt(target.id);
		await refresh();
		onSelectActive(remaining[0].id);
	}
</script>

{#if !loading && selectedProfile}
	<div class="space-y-3">
		<div class="flex items-center gap-2">
			<Select.Root
				type="single"
				value={selectedProfile.id}
				onValueChange={(value) => value && onSelectActive(value)}
			>
				<Select.Trigger class="w-full max-w-xs">
					{selectedProfile.name}
				</Select.Trigger>

				<Select.Content>
					{#each profiles as profile (profile.id)}
						<Select.Item value={profile.id} label={profile.name}>{profile.name}</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>

			<Button variant="ghost" size="icon" onclick={handleNew} aria-label="New profile">
				<Plus class={ICON_CLASS_DEFAULT} />
			</Button>

			<Button
				variant="ghost"
				size="icon"
				onclick={handleDuplicate}
				aria-label="Duplicate profile"
			>
				<Copy class={ICON_CLASS_DEFAULT} />
			</Button>

			<Button
				variant="ghost"
				size="icon"
				onclick={handleDelete}
				disabled={profiles.length <= 1}
				aria-label="Delete profile"
			>
				<Trash2 class={ICON_CLASS_DEFAULT} />
			</Button>
		</div>

		<div class="space-y-1.5">
			<Label for="system-prompt-name">Profile name</Label>

			<Input
				id="system-prompt-name"
				value={selectedProfile.name}
				onchange={(e: Event & { currentTarget: HTMLInputElement }) =>
					handleRename(e.currentTarget.value)}
			/>
		</div>

		<Textarea
			class="min-h-[10rem] w-full md:max-w-3xl"
			value={selectedProfile.content}
			oninput={(e: Event & { currentTarget: HTMLTextAreaElement }) =>
				handleContentInput(e.currentTarget.value)}
			placeholder={'The starting message that defines how the model should behave. Supports Jinja2-style templates, e.g. {{ pick("a", "b") }}.'}
		/>
	</div>
{/if}
