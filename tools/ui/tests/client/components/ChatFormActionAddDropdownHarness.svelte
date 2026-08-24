<script lang="ts">
	import ChatFormActionAddDropdown from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionAdd/ChatFormActionAddDropdown.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { setChatFormActionsContext } from '$lib/contexts';

	let {
		onPrecachePrefix,
		precachePrefixDisabled = false,
		showPrecachePrefix = false
	}: {
		onPrecachePrefix?: () => Promise<void>;
		precachePrefixDisabled?: boolean;
		showPrecachePrefix?: boolean;
	} = $props();

	// ChatFormActionAddDropdown reads everything from context rather than props,
	// so the harness has to stand in for ChatFormActions as the provider.
	setChatFormActionsContext({
		get disabled() {
			return false;
		},
		get hasAudioModality() {
			return false;
		},
		get hasVideoModality() {
			return false;
		},
		get hasVisionModality() {
			return false;
		},
		get hasMcpPromptsSupport() {
			return false;
		},
		get hasMcpResourcesSupport() {
			return false;
		},
		get onPrecachePrefix() {
			return onPrecachePrefix;
		},
		get precachePrefixDisabled() {
			return precachePrefixDisabled;
		},
		get showPrecachePrefix() {
			return showPrecachePrefix;
		}
	});
</script>

<Tooltip.Provider>
	<ChatFormActionAddDropdown />
</Tooltip.Provider>
