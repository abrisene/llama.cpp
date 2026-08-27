/**
 * LorasService - Stateless API layer for the router-side dynamic LoRA registry
 *
 * Wraps GET/POST/DELETE /router/loras (the registry, shared across all
 * models of a matching arch) and GET /lora-adapters (what is actually
 * resident on one loaded child model). No reactive state.
 */

import { API_LORAS } from '$lib/constants';
import { apiFetch, apiFetchWithParams, apiPost } from '$lib/utils';

export class LorasService {
	/**
	 * Register or update one or more LoRA adapters. The router diffs against
	 * its current state and forwards the full arch-matching set to children.
	 *
	 * @param entries - Adapters to add or update
	 * @returns Registry response, including per-entry results
	 */
	static async addLora(
		entries: ApiLoraRegistryAddRequest[]
	): Promise<ApiLoraRegistryAddResponse> {
		return apiPost<ApiLoraRegistryAddResponse>(API_LORAS.REGISTRY, entries);
	}

	/**
	 * Remove every adapter from the registry.
	 *
	 * @returns Deletion response with removal count and any per-model errors
	 */
	static async deleteAllLoras(): Promise<ApiLoraRegistryDeleteResponse> {
		return apiFetch<ApiLoraRegistryDeleteResponse>(API_LORAS.REGISTRY, {
			body: JSON.stringify({ all: true }),
			method: 'DELETE'
		});
	}

	/**
	 * Remove one or more adapters from the registry by path.
	 *
	 * @param paths - Paths of the adapters to remove
	 * @returns Deletion response with removal count and any per-model errors
	 */
	static async deleteLora(paths: string[]): Promise<ApiLoraRegistryDeleteResponse> {
		return apiFetch<ApiLoraRegistryDeleteResponse>(API_LORAS.REGISTRY, {
			body: JSON.stringify(paths.map((path) => ({ path }))),
			method: 'DELETE'
		});
	}

	/**
	 * Fetch the adapters currently resident on one loaded child model.
	 *
	 * @param modelName - Model identifier to query
	 * @returns Child adapter list
	 */
	static async listChildAdapters(modelName: string): Promise<ApiLoraChildAdapter[]> {
		return apiFetchWithParams<ApiLoraChildAdapter[]>(API_LORAS.CHILD_ADAPTERS, {
			model: modelName
		});
	}

	/**
	 * Fetch the full router-side LoRA registry.
	 *
	 * @returns Registry entries
	 */
	static async listLoras(): Promise<ApiLoraRegistryEntry[]> {
		return apiFetch<ApiLoraRegistryEntry[]>(API_LORAS.REGISTRY);
	}

	/**
	 * Enumerate .gguf adapter files under each configured --lora-root, with
	 * architecture peeked from each file's GGUF header. Non-adapter GGUFs
	 * are filtered out server-side.
	 *
	 * @returns Discovered adapters
	 */
	static async listAvailable(): Promise<ApiLoraAvailable[]> {
		return apiFetch<ApiLoraAvailable[]>(API_LORAS.AVAILABLE);
	}

	/**
	 * Update the scale of one registered adapter. Thin wrapper over addLora
	 * for the single-entry commit path (e.g. a debounced slider).
	 *
	 * @param entry - Adapter path, new scale, and optional alias
	 * @returns Registry response for the single entry
	 */
	static async updateLoraScale(
		entry: ApiLoraRegistryAddRequest
	): Promise<ApiLoraRegistryAddResponse> {
		return LorasService.addLora([entry]);
	}
}
