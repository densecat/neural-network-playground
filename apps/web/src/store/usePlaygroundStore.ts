import { create, type StoreApi, type UseBoundStore } from 'zustand';
import {
    DEFAULT_DEMAND,
    DEFAULT_EXPERIMENT_DOCUMENT,
    decodeExperimentUrl,
    encodeExperimentUrl,
    prepareExperimentDocument,
    resolveRecipe,
    validateExperimentDocument,
    type ExperimentDocumentV2,
    type ExperimentSchemaIssue,
    type PreparedExperimentDocumentV2,
    type RecipeCatalogEntry,
    type SchemaResult,
    type ValidatedExperimentDocumentV2,
    type ValidatedStandardExperimentRecipeV2,
    type VisualizationDemand,
} from '@nn-playground/shared';
import type { RecipeEditIssue, RecipeEditResult } from './recipeEdits.ts';

export interface FeaturesUI {
    canvasNetworkGraph: boolean;
    webgpuGrid: boolean;
}

export interface PreparationState {
    status: 'ready' | 'preparing' | 'error';
    requestId: number;
    issues: readonly ExperimentSchemaIssue[];
}

export type ExperimentInputSource =
    | { readonly kind: 'url'; readonly rawHash: string }
    | { readonly kind: 'file'; readonly file: File };

export type ExperimentAccessState =
    | {
        readonly status: 'ready';
        readonly prepared: PreparedExperimentDocumentV2;
    }
    | {
        readonly status: 'incompatible';
        readonly prepared: null;
        readonly source: ExperimentInputSource;
        readonly issues: readonly ExperimentSchemaIssue[];
    };

export type StoreRecipeEditResult =
    | SchemaResult<PreparedExperimentDocumentV2>
    | { ok: false; issue: RecipeEditIssue };

type PrepareExperiment = (
    value: unknown,
) => Promise<SchemaResult<PreparedExperimentDocumentV2>>;

export interface PlaygroundStoreInitialization {
    prepare?: PrepareExperiment;
    access: ExperimentAccessState;
    runRequestSource?: RunRequestSource;
}

type RunRequestSource = 'document' | 'saved-run' | 'url';

export interface PlaygroundStore {
    access: ExperimentAccessState;
    preparation: PreparationState;
    /** Changes when an explicit experiment replacement is published. */
    runRequestId: number;
    runRequestSource: RunRequestSource;
    urlSyncNotice: string | null;

    featuresUI: FeaturesUI;
    demand: VisualizationDemand;

    replaceDocument(value: unknown, source?: 'saved-run'): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    acknowledgeRuntimeDocument(prepared: PreparedExperimentDocumentV2, publishAccess?: boolean): void;
    restoreAcknowledgedDocument(prepared: PreparedExperimentDocumentV2): void;
    replaceImportedDocument(
        value: unknown,
        file: File,
    ): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    markIncompatible(
        source: ExperimentInputSource,
        issues: readonly ExperimentSchemaIssue[],
    ): void;
    startFresh(): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    editRecipe(
        edit: (recipe: ValidatedStandardExperimentRecipeV2) => RecipeEditResult,
    ): Promise<StoreRecipeEditResult>;
    editView(
        edit: (
            view: ValidatedExperimentDocumentV2['view'],
        ) => ExperimentDocumentV2['view'],
    ): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    applyRecipe(entry: RecipeCatalogEntry): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    syncToUrl(): SchemaResult<string>;
    loadFromUrl(): Promise<SchemaResult<PreparedExperimentDocumentV2>>;

    setDemand(demand: VisualizationDemand): void;
}

export type PlaygroundStoreApi = UseBoundStore<StoreApi<PlaygroundStore>>;

const DEFAULT_FEATURES_UI: FeaturesUI = {
    canvasNetworkGraph: true,
    webgpuGrid: true,
};

const VIEW_FIELDS = ['showTestData', 'discretizeOutput'] as const;

const NO_ACTIVE_DOCUMENT: readonly ExperimentSchemaIssue[] = Object.freeze([{
    code: 'invalid-field',
    path: '$',
    message: 'No compatible version-2 experiment is active',
}]);

function unavailableResult<T>(
    issues: readonly ExperimentSchemaIssue[],
): SchemaResult<T> {
    return { ok: false, issues: issues.length > 0 ? issues : NO_ACTIVE_DOCUMENT };
}

function exactCatalogEntry(entry: RecipeCatalogEntry): RecipeCatalogEntry | undefined {
    const resolved = resolveRecipe({ id: entry.id, revision: entry.revision });
    if (resolved !== entry) return undefined;
    if (resolved.recipe !== resolved.prepared.document.recipe) return undefined;
    return resolved;
}

export async function initializePlaygroundStateFromHash(
    rawHash: string,
    prepare: PrepareExperiment = prepareExperimentDocument,
): Promise<PlaygroundStoreInitialization> {
    const decoded = decodeExperimentUrl(rawHash);
    if (!decoded.ok) {
        return {
            prepare,
            access: {
                status: 'incompatible',
                prepared: null,
                source: { kind: 'url', rawHash },
                issues: decoded.issues,
            },
        };
    }

    const prepared = await prepare(decoded.value);
    if (!prepared.ok) {
        return {
            prepare,
            access: {
                status: 'incompatible',
                prepared: null,
                source: { kind: 'url', rawHash },
                issues: prepared.issues,
            },
        };
    }
    return {
        prepare,
        access: { status: 'ready', prepared: prepared.value },
        runRequestSource: rawHash ? 'url' : 'document',
    };
}

/**
 * Read the fragment from the full URL instead of `Location.hash` so a bare
 * trailing `#` remains distinguishable from a URL with no fragment.
 */
export function getRawExperimentHash(href: string): string {
    const hashIndex = href.indexOf('#');
    return hashIndex === -1 ? '' : href.slice(hashIndex);
}

export function initializePlaygroundStateFromLocation(
    location: Pick<Location, 'href'>,
    prepare: PrepareExperiment = prepareExperimentDocument,
): Promise<PlaygroundStoreInitialization> {
    return initializePlaygroundStateFromHash(
        getRawExperimentHash(location.href),
        prepare,
    );
}

export function createPlaygroundStore(
    initialization: PlaygroundStoreInitialization,
): PlaygroundStoreApi {
    const prepare = initialization.prepare ?? prepareExperimentDocument;
    const initialAccess = initialization.access;

    let nextRequestId = 0;
    let viewRevision = 0;
    const viewFieldRevisions = { showTestData: 0, discretizeOutput: 0 };
    let latestCandidateDocument: ValidatedExperimentDocumentV2 | null =
        initialAccess.status === 'ready' ? initialAccess.prepared.document : null;
    let lastSuccessfulPrepared = initialAccess.status === 'ready'
        ? initialAccess.prepared
        : null;
    let acknowledgedDocument: PreparedExperimentDocumentV2 | null = null;

    return create<PlaygroundStore>((set, get) => {
        const markIncompatible = (
            source: ExperimentInputSource,
            issues: readonly ExperimentSchemaIssue[],
        ): void => {
            const requestId = ++nextRequestId;
            latestCandidateDocument = null;
            acknowledgedDocument = null;
            const boundedIssues = issues.length > 0 ? issues : NO_ACTIVE_DOCUMENT;
            set({
                access: {
                    status: 'incompatible',
                    prepared: null,
                    source,
                    issues: boundedIssues,
                },
                preparation: { status: 'error', requestId, issues: [] },
                urlSyncNotice: null,
            });
        };

        const replace = async (
            value: unknown,
            failureSource: ExperimentInputSource | null = null,
            source: RunRequestSource = 'document',
        ): Promise<SchemaResult<PreparedExperimentDocumentV2>> => {
            const requestId = ++nextRequestId;
            const startingViewRevision = viewRevision;
            const candidateValidation = validateExperimentDocument(value);
            if (candidateValidation.ok) {
                latestCandidateDocument = candidateValidation.value;
            }

            set({
                preparation: { status: 'preparing', requestId, issues: [] },
            });
            const result = await prepare(
                candidateValidation.ok ? candidateValidation.value : value,
            );
            if (requestId !== nextRequestId) return result;

            if (!result.ok) {
                if (failureSource) {
                    latestCandidateDocument = null;
                    acknowledgedDocument = null;
                    set({
                        access: {
                            status: 'incompatible',
                            prepared: null,
                            source: failureSource,
                            issues: result.issues,
                        },
                        preparation: { status: 'error', requestId, issues: [] },
                        urlSyncNotice: null,
                    });
                } else {
                    latestCandidateDocument = lastSuccessfulPrepared?.document ?? null;
                    set({
                        preparation: { status: 'error', requestId, issues: result.issues },
                    });
                }
                return result;
            }

            let prepared = result.value;
            if (startingViewRevision !== viewRevision && lastSuccessfulPrepared) {
                const view = { ...prepared.document.view };
                for (const field of VIEW_FIELDS) {
                    if (viewFieldRevisions[field] > startingViewRevision) {
                        view[field] = lastSuccessfulPrepared.document.view[field];
                    }
                }
                prepared = { ...prepared, document: { ...prepared.document, view } };
            }
            lastSuccessfulPrepared = prepared;
            latestCandidateDocument = prepared.document;
            set({
                access: { status: 'ready', prepared },
                preparation: { status: 'ready', requestId, issues: [] },
                runRequestId: requestId,
                runRequestSource: source,
            });
            return { ok: true, value: prepared };
        };

        const editRecipe = async (
            edit: (recipe: ValidatedStandardExperimentRecipeV2) => RecipeEditResult,
        ): Promise<StoreRecipeEditResult> => {
            const base = latestCandidateDocument;
            if (!base) return unavailableResult(get().preparation.issues);
            const edited = edit(base.recipe);
            if (!edited.ok) return edited;
            return replace({ ...base, recipe: edited.recipe });
        };

        const editView = async (
            edit: (
                view: ValidatedExperimentDocumentV2['view'],
            ) => ExperimentDocumentV2['view'],
        ): Promise<SchemaResult<PreparedExperimentDocumentV2>> => {
            const { access } = get();
            if (access.status !== 'ready') return unavailableResult(access.issues);
            const current = access.prepared;
            const validation = validateExperimentDocument({
                ...current.document,
                view: edit(current.document.view),
            });
            if (!validation.ok) return validation;
            const { view } = validation.value;
            const changedFields = VIEW_FIELDS.filter((field) => view[field] !== current.document.view[field]);
            if (changedFields.length === 0) return { ok: true, value: current };
            const prepared = { ...current, document: { ...current.document, view } };
            viewRevision++;
            lastSuccessfulPrepared = prepared;
            const candidateView = { ...(latestCandidateDocument?.view ?? view) };
            for (const field of changedFields) {
                viewFieldRevisions[field] = viewRevision;
                candidateView[field] = view[field];
            }
            if (latestCandidateDocument) {
                latestCandidateDocument = { ...latestCandidateDocument, view: candidateView };
            }
            set({ access: { status: 'ready', prepared } });
            get().syncToUrl();
            return { ok: true, value: prepared };
        };

        const applyRecipe = async (
            entry: RecipeCatalogEntry,
        ): Promise<SchemaResult<PreparedExperimentDocumentV2>> => {
            const base = latestCandidateDocument;
            if (!base) return unavailableResult(get().preparation.issues);
            const resolved = exactCatalogEntry(entry);
            if (!resolved) {
                return {
                    ok: false,
                    issues: [{
                        code: 'invalid-field',
                        path: 'recipe',
                        message: 'Recipe catalog entry is not the resolved built-in identity pair',
                    }],
                };
            }
            return replace({
                kind: 'nn-playground-experiment',
                schemaVersion: 2,
                recipe: resolved.recipe,
                view: base.view,
            });
        };

        const restoreAcknowledgedDocument = (prepared: PreparedExperimentDocumentV2): void => {
            const { access, preparation } = get();
            if (access.status !== 'ready') return;
            // A display edit is independent of the worker's scientific recipe.
            const current = access.prepared;
            const restored = current.identities === prepared.identities ? current : {
                ...prepared,
                document: { ...prepared.document, view: current.document.view },
            };
            acknowledgedDocument = restored;
            lastSuccessfulPrepared = restored;
            if (preparation.status !== 'preparing') latestCandidateDocument = restored.document;
            if (restored !== current) set({ access: { status: 'ready', prepared: restored } });
        };

        return {
            access: initialAccess,
            preparation: initialAccess.status === 'ready'
                ? { status: 'ready', requestId: 0, issues: [] }
                : { status: 'error', requestId: 0, issues: [] },
            runRequestId: 0,
            runRequestSource: initialization.runRequestSource ?? 'document',
            urlSyncNotice: null,
            featuresUI: { ...DEFAULT_FEATURES_UI },
            demand: { ...DEFAULT_DEMAND },

            replaceDocument: (value, source) => replace(value, null, source),
            acknowledgeRuntimeDocument: (prepared, publishAccess = false) => {
                acknowledgedDocument = prepared;
                if (publishAccess) restoreAcknowledgedDocument(prepared);
            },
            restoreAcknowledgedDocument,
            replaceImportedDocument: (value, file) => replace(value, { kind: 'file', file }),
            markIncompatible,
            startFresh: async () => {
                const requestId = nextRequestId + 1;
                const result = await replace(DEFAULT_EXPERIMENT_DOCUMENT);
                if (result.ok) {
                    const { access, runRequestId } = get();
                    if (access.status !== 'ready' || runRequestId !== requestId) {
                        return {
                            ok: false,
                            issues: [{
                                code: 'invalid-field',
                                path: '$',
                                message: 'Start fresh was superseded by a newer experiment request.',
                            }],
                        };
                    }
                    get().syncToUrl();
                    return { ok: true, value: access.prepared };
                }
                return result;
            },
            editRecipe,
            editView,
            applyRecipe,
            syncToUrl: () => {
                const access = get().access;
                if (access.status !== 'ready') return unavailableResult(access.issues);
                try {
                    const displayed = access.prepared.document;
                    const hash = encodeExperimentUrl(acknowledgedDocument
                        ? { ...acknowledgedDocument.document, view: displayed.view }
                        : displayed);
                    window.history.replaceState(null, '', hash);
                    if (get().urlSyncNotice !== null) set({ urlSyncNotice: null });
                    return { ok: true, value: hash };
                } catch {
                    const message = 'The setup link could not be updated. Retry to update the link.';
                    if (get().urlSyncNotice !== message) set({ urlSyncNotice: message });
                    return { ok: false, issues: [{ code: 'invalid-field', path: '$', message }] };
                }
            },
            loadFromUrl: async () => {
                const raw = getRawExperimentHash(window.location.href);
                const decoded = decodeExperimentUrl(raw);
                if (!decoded.ok) {
                    markIncompatible({ kind: 'url', rawHash: raw }, decoded.issues);
                    return decoded;
                }
                return replace(decoded.value, { kind: 'url', rawHash: raw }, 'url');
            },

            setDemand: (demand) => set({ demand }),
        };
    });
}

const productionInitialization = typeof window === 'undefined'
    ? await initializePlaygroundStateFromHash('')
    : await initializePlaygroundStateFromLocation(window.location);

export const usePlaygroundStore = createPlaygroundStore({
    ...productionInitialization,
});
