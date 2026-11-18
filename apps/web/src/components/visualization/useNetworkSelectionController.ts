import { useCallback, useEffect, useMemo, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import { deriveNetworkSelectionModel, isNetworkNode, type NetworkNodeRef, type NetworkSelectionDisplayModel } from './networkSelectionModel.ts';

export interface NetworkSelectionController {
    readonly model: NetworkSelectionDisplayModel;
    readonly selectedNode: NetworkNodeRef | null;
    readonly commands: { selectNode(node: NetworkNodeRef): void; clearSelection(): void };
}

/** Ephemeral selection only: no recipe/URL/persistence or worker mutation. */
export function useNetworkSelectionState() {
    const generation = useTrainingStore((s) => s.evidenceGenerationId);
    const architecture = usePlaygroundStore((s) => {
        if (s.access.status !== 'ready') return '';
        const network = s.access.prepared.compiled.network;
        return [network.inputSize, ...network.hiddenLayers, network.outputSize].join(':');
    });
    const layers = useMemo(() => architecture ? architecture.split(':').map(Number) : [], [architecture]);
    const scope = `${generation ?? 'none'}|${architecture}`;
    const [selection, setSelection] = useState<{ scope: string; node: NetworkNodeRef } | null>(null);
    const selectedNode = selection?.scope === scope ? selection.node : null;
    useEffect(() => { setSelection(null); }, [scope]);
    const selectNode = useCallback((node: NetworkNodeRef) => {
        if (isNetworkNode(node, layers)) setSelection({ scope, node: { ...node } });
    }, [layers, scope]);
    const clearSelection = useCallback(() => setSelection(null), []);
    const commands = useMemo(() => ({ selectNode, clearSelection }), [selectNode, clearSelection]);
    return useMemo(() => ({ architecture, generation, selectedNode, commands }), [architecture, generation, selectedNode, commands]);
}

export type NetworkSelectionState = ReturnType<typeof useNetworkSelectionState>;

export function useNetworkSelectionModel({ architecture, generation, selectedNode, commands }: NetworkSelectionState, visible = true): NetworkSelectionController {
    const paramsVersion = useTrainingStore((s) => visible ? s.paramsVersion : 0);
    const gridsVersion = useTrainingStore((s) => visible ? s.neuronGridsVersion : 0);
    const model = useMemo(() => {
        void paramsVersion; void gridsVersion;
        if (!visible) return { kind: 'empty' } as const;
        const frame = getFrameBuffer();
        if (frame.weightLayout?.layerSizes.join(':') !== architecture) return { kind: 'empty' } as const;
        const ownsParameters = generation !== null && frame.parameterProvenance?.model.generationId === generation;
        const ownsGrids = generation !== null && frame.neuronGridsProvenance?.model.generationId === generation;
        return deriveNetworkSelectionModel({ ...frame,
            weights: ownsParameters ? frame.weights : null,
            biases: ownsParameters ? frame.biases : null,
            parameterProvenance: ownsParameters ? frame.parameterProvenance : null,
            neuronGrids: ownsGrids ? frame.neuronGrids : null,
            neuronGridsProvenance: ownsGrids ? frame.neuronGridsProvenance : null,
        }, selectedNode);
    }, [architecture, generation, gridsVersion, paramsVersion, selectedNode, visible]);
    return { model, selectedNode, commands };
}

export function useNetworkSelectionController(): NetworkSelectionController {
    return useNetworkSelectionModel(useNetworkSelectionState());
}
