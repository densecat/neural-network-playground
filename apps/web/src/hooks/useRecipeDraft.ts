import { useCallback, useEffect, useRef, useState } from 'react';
import { canonicalizeJson, validateExperimentDocument, type ExperimentRunRecordV2, type PreparedExperimentDocumentV2, type StandardExperimentRecipeV2 } from '@nn-playground/shared';
import { getDatasetContract, type DatasetId } from '@nn-playground/engine';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import { commitRecipeEditPrepared } from '../store/commitRecipeEdit.ts';

export type SetupTab = 'dataset' | 'network' | 'training';
type Draft = {
    base: PreparedExperimentDocumentV2;
    baseRunRequestId: number;
    recipe: StandardExperimentRecipeV2;
    text: Record<string, string>;
    baseline?: ExperimentRunRecordV2;
};
export interface AppliedVariation {
    readonly baselineId: string;
    readonly generationId: number;
    readonly recipeFingerprint: string;
}
export interface RecipeDraftController {
    recipe: StandardExperimentRecipeV2 | null;
    dirty: boolean; valid: boolean; canApply: boolean; busy: boolean; submitted: boolean; error: string | null;
    issues: readonly string[];
    variationBaseline: ExperimentRunRecordV2 | null;
    appliedVariation: AppliedVariation | null;
    value(path: string): unknown;
    commands: {
        set(path: string, value: unknown): void;
        number(path: string, text: string): void;
        dataset(dataset: DatasetId): void;
        preset(recipe: StandardExperimentRecipeV2): void;
        variation(record: ExperimentRunRecordV2): boolean;
        apply(): Promise<boolean>;
        cancel(): void;
        retry(): void;
    };
}
function clone<T>(value: T): T { return structuredClone(value); }
function read(value: unknown, path: string): unknown {
    return path.split('.').reduce<unknown>((part, key) => (part as Record<string, unknown>)?.[key], value);
}
function write(value: unknown, path: string, next: unknown) {
    const keys = path.split('.');
    const last = keys.pop()!;
    const target = keys.reduce<unknown>((part, key) => (part as Record<string, unknown>)[key], value);
    (target as Record<string, unknown>)[last] = next;
}

function isDirty(draft: Draft | null): boolean {
    return draft !== null && (draft.baseline !== undefined
        || JSON.stringify(draft.recipe) !== JSON.stringify(draft.base.document.recipe)
        || Object.entries(draft.text).some(([path, text]) => text.trim() === '' || !Number.isFinite(Number(text)) || Number(text) !== read(draft.base.document.recipe, path)));
}

function canApplyCandidate(draft: Draft | null, recipe: StandardExperimentRecipeV2): boolean {
    return isDirty(draft) && (!draft?.baseline || canonicalizeJson(recipe) !== canonicalizeJson(draft.baseline.recipe));
}

/** One session-only candidate; publication and worker acknowledgement are distinct. */
export function useRecipeDraft(): RecipeDraftController {
    const prepared = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared : null);
    const pending = useTrainingStore((s) => s.pendingConfigSource);
    const configError = useTrainingStore((s) => s.configError);
    const [draft, setDraft] = useState<Draft | null>(null);
    const draftRef = useRef<Draft | null>(null);
    const replaceDraft = useCallback((next: Draft | null) => {
        draftRef.current = next;
        setDraft(next);
    }, []);
    const [error, setError] = useState<string | null>(null);
    const [submitted, setSubmitted] = useState(false);
    const [appliedVariation, setAppliedVariation] = useState<AppliedVariation | null>(null);
    const lock = useRef(false);
    const target = useRef<{
        prepared: PreparedExperimentDocumentV2;
        runRequestId: number;
        baselineId?: string;
        previousGeneration: number | null;
    } | null>(null);
    const resolve = useRef<((success: boolean) => void) | null>(null);
    const recipe = draft?.recipe ?? prepared?.document.recipe ?? null;
    const validation = recipe && prepared ? validateExperimentDocument({ ...prepared.document, recipe }) : null;
    const issues = validation && !validation.ok ? validation.issues.map((i) => `${i.path}: ${i.message}`) : [];
    const dirty = isDirty(draft);
    const valid = validation?.ok === true;
    const busy = submitted || pending !== null;
    const canApply = !busy && validation?.ok === true && canApplyCandidate(draft, validation.value.recipe);

    const checkTarget = useCallback(() => {
        const request = target.current;
        if (!request) return;
        const ts = useTrainingStore.getState();
        const ps = usePlaygroundStore.getState();
        if (ps.runRequestId !== request.runRequestId || ps.access.status !== 'ready'
            || ps.access.prepared.identities !== request.prepared.identities) {
            target.current = null; lock.current = false; setSubmitted(false);
            setError('The active experiment changed while setup was applying. Discard this draft to reload it.');
            resolve.current?.(false); resolve.current = null;
            return;
        }
        if (ts.configError) {
            setError(ts.configError);
            resolve.current?.(false); resolve.current = null;
            return; // Retain exact submitted candidate for existing sync recovery.
        }
        const recipeFingerprint = request.prepared.identities.recipeFingerprint;
        const generationId = ts.evidenceGenerationId;
        const newGeneration = generationId !== null && generationId > (request.previousGeneration ?? 0);
        if (ts.pendingConfigSource === null && ts.trainedRecipeFingerprint === recipeFingerprint
            && ts.trainedRecipeSource === 'config-sync' && (!request.baselineId || newGeneration)) {
            target.current = null; lock.current = false;
            setAppliedVariation(request.baselineId && generationId !== null
                ? { baselineId: request.baselineId, generationId, recipeFingerprint }
                : null);
            replaceDraft(null); setSubmitted(false); setError(null);
            resolve.current?.(true); resolve.current = null;
        }
    }, [replaceDraft]);
    useEffect(() => {
        const unsubscribe = useTrainingStore.subscribe(checkTarget);
        const unsubscribePlayground = usePlaygroundStore.subscribe(checkTarget);
        return () => { unsubscribe(); unsubscribePlayground(); resolve.current?.(false); };
    }, [checkTarget]);
    useEffect(() => {
        if (!dirty) return;
        const guard = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
        window.addEventListener('beforeunload', guard);
        return () => window.removeEventListener('beforeunload', guard);
    }, [dirty]);
    const edit = (path: string, value: unknown, text?: string) => {
        if (lock.current || !prepared) return;
        const current = draftRef.current;
        const next: Draft = current ? { ...current, recipe: clone(current.recipe), text: { ...current.text } }
            : { base: prepared, baseRunRequestId: usePlaygroundStore.getState().runRequestId, recipe: clone(prepared.document.recipe), text: {} };
        write(next.recipe, path, value);
        for (const key of Object.keys(next.text)) if (key === path || key.startsWith(`${path}.`)) delete next.text[key];
        if (text !== undefined) next.text[path] = text;
        replaceDraft(next); setError(null);
    };
    return {
        recipe, dirty, valid, canApply, busy, submitted, error: error ?? configError, issues,
        variationBaseline: draft?.baseline ?? null, appliedVariation,
        value: (path) => draft?.text[path] ?? read(recipe, path),
        commands: {
            set: edit,
            preset: (recipe) => {
                if (lock.current || pending !== null || !prepared) return;
                const next: Draft = {
                    base: draftRef.current?.base ?? prepared,
                    baseRunRequestId: draftRef.current?.baseRunRequestId ?? usePlaygroundStore.getState().runRequestId,
                    recipe: clone(recipe),
                    text: {},
                    baseline: draftRef.current?.baseline,
                };
                replaceDraft(next);
                setError(null);
            },
            variation: (record) => {
                const state = usePlaygroundStore.getState();
                if (lock.current || isDirty(draftRef.current) || state.access.status !== 'ready'
                    || state.preparation.status === 'preparing' || useTrainingStore.getState().pendingConfigSource !== null) return false;
                replaceDraft({ base: state.access.prepared, baseRunRequestId: state.runRequestId, recipe: clone(record.recipe), text: {}, baseline: record });
                setError(null);
                return true;
            },
            number: (path, text) => edit(path, text.trim() === '' ? NaN : Number(text), text),
            dataset: (dataset) => {
                if ((draftRef.current?.recipe ?? prepared?.document.recipe)?.task.dataset === dataset) return;
                const kind = getDatasetContract(dataset).taskKind;
                edit('task', { kind, dataset });
                edit('objective.dataLoss', { kind: kind === 'regression' ? 'mean-squared-error' : kind === 'multiclass-classification' ? 'categorical-cross-entropy-with-logits' : 'binary-cross-entropy-with-logits' });
            },
            cancel: () => {
                if (lock.current) return;
                replaceDraft(null); setError(null);
            },
            retry: () => useTrainingStore.getState().retryConfigSync(),
            apply: async () => {
                const candidate = draftRef.current;
                if (lock.current || !candidate || useTrainingStore.getState().pendingConfigSource !== null) return false;
                const ps = usePlaygroundStore.getState();
                if (ps.access.status !== 'ready' || ps.runRequestId !== candidate.baseRunRequestId
                    || ps.access.prepared.identities.canonicalRecipeKey !== candidate.base.identities.canonicalRecipeKey) {
                    setError('The active recipe changed. Cancel to reload it before applying setup.'); return false;
                }
                const validation = validateExperimentDocument({ ...ps.access.prepared.document, recipe: candidate.recipe });
                if (!validation.ok || !canApplyCandidate(candidate, validation.value.recipe)) return false;
                const previousGeneration = useTrainingStore.getState().evidenceGenerationId;
                lock.current = true; setSubmitted(true); setError(null);
                const published = await commitRecipeEditPrepared('setup', () => ({ ok: true, recipe: candidate.recipe }), candidate.base.identities.canonicalRecipeKey);
                if (!published) {
                    lock.current = false; setSubmitted(false);
                    setError(useTrainingStore.getState().configError ?? 'Setup could not be applied. Your draft is preserved.');
                    return false;
                }
                const { access, runRequestId } = usePlaygroundStore.getState();
                if (access.status !== 'ready' || access.prepared.identities !== published.identities) {
                    lock.current = false; setSubmitted(false);
                    setError('The active experiment changed while setup was applying. Discard this draft to reload it.');
                    return false;
                }
                target.current = { prepared: published, runRequestId, baselineId: candidate.baseline?.id, previousGeneration };
                return new Promise<boolean>((done) => {
                    resolve.current = done;
                    checkTarget();
                });
            },
        },
    };
}
