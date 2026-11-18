import { useCallback, useEffect, useRef, useState } from 'react';
import { EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS } from '@nn-playground/shared';
import { useExperimentMemoryStore } from '../store/experimentMemoryStore.ts';
import { captureCurrentRun } from './captureCurrentRun.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import type { AppliedVariation } from './useRecipeDraft.ts';

interface VariationComparison {
    readonly baselineId: string;
    readonly variationId: string;
}

type PendingVariationComparison = VariationComparison & { readonly generationId: number };

/** Shared by the transport and History; App owns the production instance. */
export interface SaveCurrentRunController {
    readonly busy: boolean;
    readonly error: string | null;
    readonly disabledReason: string | null;
    readonly pending: boolean;
    readonly comparisonOffer?: VariationComparison | null;
    readonly commands: {
        readonly save: (title?: string) => Promise<boolean>;
        readonly retry: () => Promise<boolean>;
        readonly discard: () => Promise<void>;
        readonly downloadPending: () => Promise<boolean>;
        readonly dismiss: () => void;
    };
}

function saveDisabledReason(busy: boolean): string | null {
    const memory = useExperimentMemoryStore.getState();
    if (busy) return 'A save operation is in progress.';
    if (memory.pendingSave) return 'Retry or discard the pending artifact before saving another run.';
    if (memory.hydrationStatus !== 'ready') return 'Saved runs are loading.';
    if (usePlaygroundStore.getState().access.status !== 'ready') return 'A compatible experiment is required.';
    return null;
}

export function useSaveCurrentRun(variation: AppliedVariation | null = null): SaveCurrentRunController {
    // Subscribe to display changes; command guards read current stores synchronously
    // so two surfaces cannot race between an event and the next React render.
    const hydrationStatus = useExperimentMemoryStore((state) => state.hydrationStatus);
    const pendingSave = useExperimentMemoryStore((state) => state.pendingSave);
    const persistenceError = useExperimentMemoryStore((state) => state.persistenceError);
    const accessStatus = usePlaygroundStore((state) => state.access.status);
    const [busy, setBusy] = useState(false);
    const [actionError, setActionError] = useState<string | null>(null);
    const [comparisonOffer, setComparisonOffer] = useState<VariationComparison | null>(null);
    const pendingComparison = useRef<PendingVariationComparison | null>(null);
    const offeredGeneration = useRef<number | null>(null);
    const busyRef = useRef(false);
    const mounted = useRef(true);
    useEffect(() => {
        mounted.current = true;
        return () => { mounted.current = false; };
    }, []);

    const operate = useCallback(async (action: () => Promise<boolean>): Promise<boolean> => {
        if (busyRef.current) return false;
        busyRef.current = true;
        if (mounted.current) { setBusy(true); setActionError(null); }
        try {
            return await action();
        } catch (error) {
            if (mounted.current) setActionError(error instanceof Error ? error.message : 'Saving the current run failed.');
            return false;
        } finally {
            busyRef.current = false;
            if (mounted.current) setBusy(false);
        }
    }, []);

    const completeComparison = useCallback((comparison: PendingVariationComparison | null) => {
        if (!comparison) return;
        offeredGeneration.current = comparison.generationId;
        const { baselineId, variationId } = comparison;
        if (mounted.current) setComparisonOffer({ baselineId, variationId });
    }, []);

    const save = useCallback(async (title = '') => {
        const trimmed = title.trim();
        if (saveDisabledReason(busyRef.current)
            || Array.from(trimmed).length > EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS) return false;
        const timestamp = new Date().toISOString();
        return operate(async () => {
            const artifact = await captureCurrentRun(trimmed, timestamp);
            const comparison = variation && offeredGeneration.current !== variation.generationId
                && artifact.snapshot.model.generationId === variation.generationId
                && artifact.recipeFingerprint === variation.recipeFingerprint
                ? { baselineId: variation.baselineId, variationId: artifact.id, generationId: variation.generationId }
                : null;
            const saved = await useExperimentMemoryStore.getState().saveRecord(artifact);
            pendingComparison.current = !saved && useExperimentMemoryStore.getState().pendingSave?.id === artifact.id
                ? comparison
                : null;
            if (saved) completeComparison(comparison);
            return saved;
        });
    }, [completeComparison, operate, variation]);
    const retry = useCallback(() => operate(async () => {
        // Never contact the worker, mint an ID, or read the current recipe here.
        const pendingId = useExperimentMemoryStore.getState().pendingSave?.id;
        const saved = await useExperimentMemoryStore.getState().retryPersistence();
        if (saved) {
            if (pendingComparison.current?.variationId === pendingId) completeComparison(pendingComparison.current);
            pendingComparison.current = null;
        }
        return saved;
    }), [completeComparison, operate]);
    const discard = useCallback(async () => {
        await operate(async () => {
            await useExperimentMemoryStore.getState().discardPendingSave();
            pendingComparison.current = null;
            return true;
        });
    }, [operate]);
    const downloadPending = useCallback(() => operate(async () => {
        const artifact = useExperimentMemoryStore.getState().pendingSave;
        if (!artifact) return false;
        // Serialize this exact retained artifact; never consult the active experiment.
        const url = URL.createObjectURL(new Blob([JSON.stringify(artifact)], { type: 'application/json' }));
        try {
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `${artifact.id}.json`;
            anchor.click();
        } finally { URL.revokeObjectURL(url); }
        return true;
    }), [operate]);
    const dismiss = useCallback(() => {
        setActionError(null);
        useExperimentMemoryStore.getState().dismissPersistenceError();
    }, []);

    void hydrationStatus;
    void accessStatus;
    return {
        busy, pending: pendingSave !== null, comparisonOffer,
        error: actionError ?? persistenceError?.message ?? null,
        disabledReason: saveDisabledReason(busy),
        commands: { save, retry, discard, downloadPending, dismiss },
    };
}
