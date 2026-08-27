<script lang="ts">
	import { cn, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';
	import { Slider as SliderPrimitive } from 'bits-ui';

	let {
		class: className,
		max = 100,
		min = 0,
		ref = $bindable(null),
		step = 1,
		value = $bindable(),
		...restProps
	}: WithoutChildrenOrChild<SliderPrimitive.RootProps> = $props();
</script>

{#snippet Thumb({ index }: { index: number })}
	<SliderPrimitive.Thumb
		{index}
		data-slot="slider-thumb"
		class="block size-4 shrink-0 rounded-full border border-primary bg-background shadow-sm ring-ring/50 transition-[color,box-shadow] hover:ring-4 focus-visible:ring-4 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50"
	/>
{/snippet}

<SliderPrimitive.Root
	bind:ref
	bind:value={value as never}
	data-slot="slider"
	{min}
	{max}
	{step}
	class={cn(
		'relative flex w-full touch-none items-center select-none data-[disabled]:opacity-50 data-[orientation=vertical]:h-full data-[orientation=vertical]:min-h-44 data-[orientation=vertical]:w-auto data-[orientation=vertical]:flex-col',
		className
	)}
	{...restProps}
>
	{#snippet children({ thumbs })}
		<span
			data-slot="slider-track"
			class="relative grow overflow-hidden rounded-full bg-muted data-[orientation=horizontal]:h-1.5 data-[orientation=horizontal]:w-full data-[orientation=vertical]:h-full data-[orientation=vertical]:w-1.5"
		>
			<SliderPrimitive.Range
				data-slot="slider-range"
				class="absolute bg-primary data-[orientation=horizontal]:h-full data-[orientation=vertical]:w-full"
			/>
		</span>

		{#each thumbs as index (index)}
			{@render Thumb({ index })}
		{/each}
	{/snippet}
</SliderPrimitive.Root>
