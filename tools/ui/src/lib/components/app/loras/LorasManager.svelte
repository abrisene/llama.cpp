<script lang="ts">
	import { ChevronDown, Layers, Plus, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as Empty from '$lib/components/ui/empty';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Select from '$lib/components/ui/select';
	import { Slider } from '$lib/components/ui/slider';
	import * as Table from '$lib/components/ui/table';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { ServerModelsSseEventType, ServerModelStatus } from '$lib/enums';
	import { LorasService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import { debounce } from '$lib/utils';
	import { onDestroy, onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	const PROMPT_PREFIX_TRUNCATE_LENGTH = 40;
	const SCALE_STEP = 0.05;
	const SCALE_MIN = 0;
	const SCALE_MAX = 1;
	const SCALE_DEBOUNCE_MS = 300;
	const DEFAULT_INITIAL_SCALE = 0.8;

	function basename(path: string): string {
		const parts = path.split(/[/\\]/);

		return parts[parts.length - 1] || path;
	}

	function truncate(text: string, length: number): string {
		if (text.length <= length) return text;

		return `${text.slice(0, length)}…`;
	}

	/**
	 *
	 *
	 * Registry (top section)
	 *
	 *
	 */

	let registry = $state<ApiLoraRegistryEntry[]>([]);
	let registryLoading = $state(false);
	let registryError = $state<string | null>(null);

	// local per-row slider draft values, so the thumb tracks input before the
	// debounced commit lands and the next fetch overwrites it
	let scaleDrafts = $state<Record<string, number>>({});

	let sortedRegistry = $derived(
		[...registry].sort((a, b) => a.arch.localeCompare(b.arch) || a.path.localeCompare(b.path))
	);

	async function refreshRegistry(): Promise<void> {
		registryLoading = true;
		registryError = null;

		try {
			registry = await LorasService.listLoras();
		} catch (error) {
			registryError = error instanceof Error ? error.message : 'Failed to load LoRA registry';
		} finally {
			registryLoading = false;
		}
	}

	function draftScale(entry: ApiLoraRegistryEntry): number {
		return scaleDrafts[entry.path] ?? entry.scale;
	}

	const commitScale = debounce(async (entry: ApiLoraRegistryEntry, scale: number) => {
		try {
			const response = await LorasService.updateLoraScale({
				alias: entry.alias,
				path: entry.path,
				scale
			});

			const result = response.results?.[0];

			if (result && !result.success) {
				toast.error(result.error || response.error || 'Failed to update scale');
			}

			await refreshRegistry();
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to update scale');
		} finally {
			delete scaleDrafts[entry.path];
		}
	}, SCALE_DEBOUNCE_MS);

	function handleScaleInput(entry: ApiLoraRegistryEntry, value: number) {
		scaleDrafts[entry.path] = value;
		commitScale(entry, value);
	}

	async function handleDelete(entry: ApiLoraRegistryEntry) {
		const label = entry.alias || basename(entry.path);
		const confirmed = window.confirm(`Remove LoRA adapter "${label}" from the registry?`);

		if (!confirmed) return;

		try {
			const response = await LorasService.deleteLora([entry.path]);

			if (response.removed === 0) {
				toast.error('Failed to remove adapter');
			}

			await refreshRegistry();
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to remove adapter');
		}
	}

	/**
	 *
	 *
	 * Add adapter (middle section)
	 *
	 *
	 */

	let addOpen = $state(false);
	let addPath = $state('');
	let addAlias = $state('');
	let addScale = $state(DEFAULT_INITIAL_SCALE);
	let addSubmitting = $state(false);
	let addError = $state<string | null>(null);
	let available = $state<ApiLoraAvailable[]>([]);
	let availableLoading = $state(false);
	let availableError = $state<string | null>(null);

	async function refreshAvailable(): Promise<void> {
		availableLoading = true;
		availableError = null;
		try {
			available = await LorasService.listAvailable();
		} catch (error) {
			availableError = error instanceof Error ? error.message : 'Failed to list available adapters';
		} finally {
			availableLoading = false;
		}
	}

	const registeredPaths = $derived(new Set(registry.map((e) => e.path)));

	const availableGrouped = $derived.by(() => {
		const groups = new Map<string, ApiLoraAvailable[]>();
		for (const a of available) {
			if (!groups.has(a.arch)) groups.set(a.arch, []);
			groups.get(a.arch)!.push(a);
		}
		return [...groups.entries()]
			.sort(([a], [b]) => a.localeCompare(b))
			.map(([arch, items]) => ({
				arch,
				items: items.sort((x, y) => x.filename.localeCompare(y.filename))
			}));
	});

	function resetAddForm() {
		addPath = '';
		addAlias = '';
		addScale = DEFAULT_INITIAL_SCALE;
		addError = null;
	}

	async function handleAddSubmit(event: SubmitEvent) {
		event.preventDefault();

		const path = addPath.trim();

		if (!path) return;

		addSubmitting = true;
		addError = null;

		try {
			const response = await LorasService.addLora([
				{
					alias: addAlias.trim() || undefined,
					path,
					scale: addScale
				}
			]);

			const result = response.results?.[0];

			if (result && !result.success) {
				addError = result.error || response.error || 'Failed to add adapter';

				return;
			}

			if (!result && response.error) {
				addError = response.error;

				return;
			}

			resetAddForm();
			await refreshRegistry();
		} catch (error) {
			addError = error instanceof Error ? error.message : 'Failed to add adapter';
		} finally {
			addSubmitting = false;
		}
	}

	/**
	 *
	 *
	 * Currently-resident on model (bottom section)
	 *
	 *
	 */

	let selectedModelId = $state<string | null>(null);
	let childAdapters = $state<ApiLoraChildAdapter[]>([]);
	let childLoading = $state(false);
	let childError = $state<string | null>(null);

	let models = $derived(modelsStore.routerModels);

	let selectedModelStatus = $derived(
		models.find((m) => m.id === selectedModelId)?.status.value ?? null
	);

	let selectedModelLoaded = $derived(
		selectedModelStatus === ServerModelStatus.LOADED ||
			selectedModelStatus === ServerModelStatus.SLEEPING
	);

	$effect(() => {
		if (selectedModelId !== null) return;

		const active = modelsStore.activeModelId;

		if (active) selectedModelId = active;
	});

	$effect(() => {
		if (models.length === 0) {
			void modelsStore.fetchRouterModels();
		}
	});

	async function refreshChildAdapters() {
		if (!selectedModelId || !selectedModelLoaded) {
			childAdapters = [];

			return;
		}

		childLoading = true;
		childError = null;

		try {
			childAdapters = await LorasService.listChildAdapters(selectedModelId);
		} catch (error) {
			childError = error instanceof Error ? error.message : 'Failed to load resident adapters';
			childAdapters = [];
		} finally {
			childLoading = false;
		}
	}

	$effect(() => {
		void selectedModelId;
		void selectedModelLoaded;

		void refreshChildAdapters();
	});

	function registryEntryForPath(path: string): ApiLoraRegistryEntry | null {
		return registry.find((entry) => entry.path === path) ?? null;
	}

	function isArchMismatch(adapter: ApiLoraChildAdapter): boolean {
		const entry = registryEntryForPath(adapter.path);

		if (!entry || !selectedModelId) return false;

		return !entry.applied_to.includes(selectedModelId);
	}

	function childAdapterLabel(adapter: ApiLoraChildAdapter): string {
		const entry = registryEntryForPath(adapter.path);

		return entry?.alias || basename(adapter.path);
	}

	/**
	 *
	 *
	 * Refresh triggers: window focus, router SSE models_reload, mutations
	 *
	 *
	 */

	function handleWindowFocus() {
		void refreshRegistry();
		void refreshChildAdapters();
	}

	let sseController: AbortController | null = null;

	onMount(() => {
		void refreshRegistry();
		void refreshAvailable();

		sseController = new AbortController();

		void ModelsService.watchModelEvents(sseController.signal, (event) => {
			if (event.event === ServerModelsSseEventType.MODELS_RELOAD) {
				void refreshRegistry();
				void refreshChildAdapters();
			}
		});
	});

	onDestroy(() => {
		sseController?.abort();
	});
</script>

<svelte:window onfocus={handleWindowFocus} />

<div class="space-y-8 {className}">
	<div class="flex items-center gap-2">
		<Layers class="h-5 w-5 md:h-6 md:w-6" />

		<h1 class="text-lg font-semibold md:text-2xl">LoRA Adapters</h1>
	</div>

	<!-- Registry -->
	<section class="space-y-3">
		<h2 class="text-sm font-medium text-muted-foreground">Registry</h2>

		{#if registryError}
			<p class="text-sm text-destructive">{registryError}</p>
		{/if}

		{#if !registryLoading && sortedRegistry.length === 0}
			<Empty.Root class="border">
				<Empty.Header>
					<Empty.Media variant="icon">
						<Layers />
					</Empty.Media>

					<Empty.Title>No LoRA adapters registered</Empty.Title>

					<Empty.Description>No LoRA adapters registered. Add one below.</Empty.Description>
				</Empty.Header>
			</Empty.Root>
		{:else}
			<div class="overflow-x-auto rounded-md border">
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head>Alias</Table.Head>
							<Table.Head>Arch</Table.Head>
							<Table.Head class="w-[14rem]">Scale</Table.Head>
							<Table.Head>Applied to</Table.Head>
							<Table.Head class="w-[3rem]"></Table.Head>
						</Table.Row>
					</Table.Header>

					<Table.Body>
						{#each sortedRegistry as entry (entry.path)}
							<Table.Row>
								<Table.Cell class="font-medium">{entry.alias || basename(entry.path)}</Table.Cell>

								<Table.Cell>
									<span
										class="inline-flex items-center rounded-md border bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
									>
										{entry.arch}
									</span>
								</Table.Cell>

								<Table.Cell>
									<div class="flex items-center gap-2">
										<Slider
											type="single"
											value={draftScale(entry)}
											min={SCALE_MIN}
											max={SCALE_MAX}
											step={SCALE_STEP}
											class="w-24"
											onValueChange={(value) => handleScaleInput(entry, value)}
										/>

										<span class="w-10 text-right text-xs text-muted-foreground tabular-nums">
											{draftScale(entry).toFixed(2)}
										</span>
									</div>
								</Table.Cell>

								<Table.Cell class="text-sm text-muted-foreground">
									{entry.applied_to.length > 0 ? entry.applied_to.join(', ') : '—'}
								</Table.Cell>

								<Table.Cell>
									<Button
										variant="ghost"
										size="icon"
										aria-label="Remove adapter"
										onclick={() => handleDelete(entry)}
									>
										<Trash2 class={ICON_CLASS_DEFAULT} />
									</Button>
								</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			</div>
		{/if}
	</section>

	<!-- Add adapter -->
	<Collapsible.Root bind:open={addOpen} class="space-y-3">
		<Collapsible.Trigger
			class="flex w-full cursor-pointer items-center justify-between gap-2 rounded-md border px-4 py-2 text-left"
		>
			<span class="text-sm font-medium">Add adapter</span>

			<ChevronDown
				class="h-4 w-4 shrink-0 transition-transform duration-200 {addOpen ? 'rotate-180' : ''}"
			/>
		</Collapsible.Trigger>

		<Collapsible.Content>
			<form onsubmit={handleAddSubmit} class="space-y-4 rounded-md border p-4">
				<div class="space-y-1.5">
					<div class="flex items-center justify-between">
						<Label for="lora-add-path">Adapter</Label>

						<Button
							type="button"
							variant="ghost"
							size="sm"
							onclick={() => void refreshAvailable()}
							disabled={availableLoading}
						>
							Rescan
						</Button>
					</div>

					<Select.Root
						type="single"
						value={addPath || undefined}
						onValueChange={(value) => (addPath = value ?? '')}
					>
						<Select.Trigger id="lora-add-path" class="w-full">
							{#if addPath}
								{available.find((a) => a.path === addPath)?.filename ?? basename(addPath)}
							{:else if availableLoading}
								Loading available adapters…
							{:else if available.length === 0}
								No adapters found under any --lora-root
							{:else}
								Select an adapter
							{/if}
						</Select.Trigger>

						<Select.Content>
							{#each availableGrouped as group (group.arch)}
								<Select.Group>
									<Select.GroupHeading>{group.arch}</Select.GroupHeading>

									{#each group.items as item (item.path)}
										{@const registered = registeredPaths.has(item.path)}

										<Select.Item value={item.path} label={item.filename} disabled={registered}>
											<div class="flex w-full items-center justify-between gap-4">
												<span class="truncate">{item.filename}</span>

												<span class="shrink-0 text-xs text-muted-foreground tabular-nums">
													{item.size_mb.toFixed(0)} MB{registered ? ' · registered' : ''}
												</span>
											</div>
										</Select.Item>
									{/each}
								</Select.Group>
							{/each}
						</Select.Content>
					</Select.Root>

					{#if availableError}
						<p class="text-xs text-destructive">{availableError}</p>
					{:else}
						<p class="text-xs text-muted-foreground">
							Discovered under the server's --lora-root directories. Grouped by adapter architecture.
						</p>
					{/if}
				</div>

				<div class="space-y-1.5">
					<Label for="lora-add-alias">Alias (optional)</Label>

					<Input id="lora-add-alias" bind:value={addAlias} placeholder="my-adapter" />
				</div>

				<div class="space-y-1.5">
					<Label for="lora-add-scale">Initial scale</Label>

					<div class="flex items-center gap-2">
						<Slider
							id="lora-add-scale"
							type="single"
							bind:value={addScale}
							min={SCALE_MIN}
							max={SCALE_MAX}
							step={SCALE_STEP}
							class="w-40"
						/>

						<span class="w-10 text-right text-xs text-muted-foreground tabular-nums">
							{addScale.toFixed(2)}
						</span>
					</div>
				</div>

				{#if addError}
					<p class="text-sm text-destructive">{addError}</p>
				{/if}

				<Button type="submit" size="sm" disabled={addSubmitting || !addPath.trim()}>
					<Plus class="h-4 w-4" />
					Add adapter
				</Button>
			</form>
		</Collapsible.Content>
	</Collapsible.Root>

	<!-- Currently-resident on model -->
	<section class="space-y-3">
		<h2 class="text-sm font-medium text-muted-foreground">Currently resident on model</h2>

		<Select.Root
			type="single"
			value={selectedModelId ?? undefined}
			onValueChange={(value) => (selectedModelId = value || null)}
		>
			<Select.Trigger class="w-full md:w-80">
				{models.find((m) => m.id === selectedModelId)?.id ?? 'Select a model'}
			</Select.Trigger>

			<Select.Content>
				{#each models as model (model.id)}
					<Select.Item value={model.id} label={model.id}>{model.id}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		{#if !selectedModelId}
			<p class="text-sm text-muted-foreground">Select a model to see its resident adapters.</p>
		{:else if !selectedModelLoaded}
			<p class="text-sm text-muted-foreground">Model not loaded</p>
		{:else if childError}
			<p class="text-sm text-destructive">{childError}</p>
		{:else if !childLoading && childAdapters.length === 0}
			<p class="text-sm text-muted-foreground">No adapters resident on this model.</p>
		{:else}
			<div class="overflow-x-auto rounded-md border">
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head class="w-[4rem]">ID</Table.Head>
							<Table.Head>Alias</Table.Head>
							<Table.Head class="w-[6rem]">Scale</Table.Head>
							<Table.Head>Task name</Table.Head>
							<Table.Head>Prompt prefix</Table.Head>
						</Table.Row>
					</Table.Header>

					<Table.Body>
						{#each childAdapters as adapter (adapter.id)}
							{@const mismatch = isArchMismatch(adapter)}

							<Table.Row class={mismatch ? 'bg-destructive/10' : ''}>
								<Table.Cell>{adapter.id}</Table.Cell>
								<Table.Cell class="font-medium">{childAdapterLabel(adapter)}</Table.Cell>
								<Table.Cell>{adapter.scale.toFixed(2)}</Table.Cell>
								<Table.Cell>{adapter.task_name || '—'}</Table.Cell>

								<Table.Cell>
									{#if adapter.prompt_prefix && adapter.prompt_prefix.length > PROMPT_PREFIX_TRUNCATE_LENGTH}
										<Tooltip.Root>
											<Tooltip.Trigger class="text-left">
												{truncate(adapter.prompt_prefix, PROMPT_PREFIX_TRUNCATE_LENGTH)}
											</Tooltip.Trigger>

											<Tooltip.Content class="max-w-sm">
												<p>{adapter.prompt_prefix}</p>
											</Tooltip.Content>
										</Tooltip.Root>
									{:else}
										{adapter.prompt_prefix || '—'}
									{/if}
								</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			</div>
		{/if}
	</section>
</div>
