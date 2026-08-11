<script lang="ts">
	import ChatFormActionAddDropdown from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionAdd/ChatFormActionAddDropdown.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let {
		onPrecachePrefix,
		precachePrefixDisabled = false,
		showPrecachePrefix = false
	}: {
		onPrecachePrefix?: () => Promise<void>;
		precachePrefixDisabled?: boolean;
		showPrecachePrefix?: boolean;
	} = $props();
</script>

<Tooltip.Provider>
	<ChatFormActionAddDropdown
		{onPrecachePrefix}
		{precachePrefixDisabled}
		{showPrecachePrefix}
	/>
</Tooltip.Provider>
