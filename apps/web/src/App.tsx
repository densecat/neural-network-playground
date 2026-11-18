// App owns model lifetime, recipe drafts and visualization demand across navigation.
import { lazy, memo, Suspense, useEffect, useRef, useCallback, useState, type ComponentType, type ReactNode } from 'react';
import { DEFAULT_DEMAND, type ExperimentRunRecordV2 } from '@nn-playground/shared';
import { useLayoutStore } from './store/useLayoutStore.ts';
import { useExperimentMemoryStore } from './store/experimentMemoryStore.ts';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { selectCheckpointBranchComplete } from './store/evidenceSelectors.ts';
import { usePlaygroundStore } from './store/usePlaygroundStore.ts';
import { useThemeEffect } from './store/theme.ts';
import { useSaveCurrentRun } from './hooks/useSaveCurrentRun.ts';
import { useTraining } from './hooks/useTraining.ts';
import { useAtelierViewport } from './hooks/useAtelierViewport.ts';
import { FIRST_RUN_EXERCISE, useRecipeDraft } from './hooks/useRecipeDraft.ts';
import { useExperimentMemoryStorageSync } from './hooks/useExperimentMemoryStorageSync.ts';
import { useNetworkSelectionModel, useNetworkSelectionState, type NetworkSelectionState } from './components/visualization/useNetworkSelectionController.ts';
import { useDecisionBoundaryController, useDecisionOverlayPreference, type DecisionOverlayPreference } from './components/visualization/useDecisionBoundaryController.ts';
import { AtelierHeader } from './components/atelier/AtelierHeader.tsx';
import { AtelierNetwork, type AtelierNetworkProps, type NetworkRegion } from './components/atelier/AtelierNetwork.tsx';
import { AtelierPrediction, DatasetPlotStatus, selectDatasetPlotStatus } from './components/atelier/AtelierPrediction.tsx';
import { AtelierTransport, EvidenceMetrics, getTrainingUnavailableReason } from './components/atelier/AtelierTransport.tsx';
import { SetupEditor } from './components/atelier/setup/SetupEditor.tsx';
import { CheckpointPanel } from './components/atelier/CheckpointPanel.tsx';
import { Dialog, TabPanels, Tabs } from './components/atelier/ui.tsx';
import { AccessibilityAnnouncer } from './components/layout/AccessibilityAnnouncer.tsx';
import { CompatibilityState } from './components/common/CompatibilityState.tsx';
import { ErrorBoundary } from './components/common/ErrorBoundary.tsx';
import { CurrentRunCard } from './components/controls/CurrentRunCard.tsx';
import type { LessonTarget } from './lessons/lessonRegistry.ts';
import type { Destination, WorkspaceTab, SetupTab, ResultsTab, UtilitySurface } from './productShell/atelierTypes.ts';
import { resolveTrainingShortcut, TRAINING_SHORTCUTS } from './shortcuts/trainingShortcuts.ts';

class ViewLoadError extends Error {}

function lazyView<Props>(load: () => Promise<ComponentType<Props>>) {
    return lazy(async () => {
        try {
            return { default: await load() };
        } catch (cause) {
            throw new ViewLoadError('The code for this view could not be loaded.', { cause });
        }
    });
}

const ConceptLibrary = lazyView(() => import('./components/EducationContent.ts').then((module) => module.ConceptLibrary));
const GuidedLessonPanel = lazyView(() => import('./components/EducationContent.ts').then((module) => module.GuidedLessonPanel));
const TrainingExplanationPanel = lazyView(() => import('./components/EducationContent.ts').then((module) => module.TrainingExplanationPanel));
const RunHistoryPanel = lazyView(() => import('./components/WorkspaceUtilities.ts').then((module) => module.RunHistoryPanel));
const ConfigPanel = lazyView(() => import('./components/WorkspaceUtilities.ts').then((module) => module.ConfigPanel));
const CodeExportPanel = lazyView(() => import('./components/controls/CodeExportPanel.tsx').then((module) => module.CodeExportPanel));
const InspectionPanel = lazyView(() => import('./components/controls/InspectionPanel.tsx').then((module) => module.InspectionPanel));
const LossChart = lazyView(() => import('./components/ResultsContent.ts').then((module) => module.LossChart));
const ConfusionMatrix = lazyView(() => import('./components/ResultsContent.ts').then((module) => module.ConfusionMatrix));
const WORKSPACE_TABS: { id: WorkspaceTab; label: string }[] = [{ id:'setup',label:'Setup' },{ id:'network',label:'Network' },{ id:'results',label:'Results' },{ id:'inspect',label:'Inspect' }];
const RESULTS_TABS: { id: ResultsTab; label: string }[] = [{ id:'boundary',label:'Prediction' },{ id:'learning',label:'Learning progress' },{ id:'errors',label:'Errors & confusion' }];
const REGRESSION_RESULTS_TABS = RESULTS_TABS.filter(({ id }) => id !== 'errors');
const EXPORT_TABS = [{ id: 'setup', label: 'Setup & sharing' }, { id: 'code', label: 'Code' }] as const;
const loading = <p role="status">Loading workspace…</p>;
const canRetryView = (error: Error) => !(error instanceof ViewLoadError);

function OptionalView({ title, children }: { title: string; children: ReactNode }) {
    return <ErrorBoundary title={title} description="Choose another view to continue your experiment." canRetry={canRetryView}>
        <Suspense fallback={loading}>{children}</Suspense>
    </ErrorBoundary>;
}

export default function App() {
    useThemeEffect();
    useExperimentMemoryStorageSync();
    const access = usePlaygroundStore((state) => state.access);
    const startFresh = usePlaygroundStore((state) => state.startFresh);
    useEffect(() => {
        const loadLocation = () => { void usePlaygroundStore.getState().loadFromUrl(); };
        window.addEventListener('hashchange', loadLocation);
        return () => window.removeEventListener('hashchange', loadLocation);
    }, []);
    const recoverWithDefault = useCallback(async () => {
        const result = await startFresh();
        if (!result.ok) throw new Error(result.issues.map((issue) => issue.message).join(' '));
    }, [startFresh]);
    if (access.status === 'incompatible') return <CompatibilityState access={access} onStartFresh={recoverWithDefault} />;
    return <AtelierShell />;
}

function AtelierShell() {
    const viewport = useAtelierViewport();
    const [networkRegion, setNetworkRegion] = useState<NetworkRegion>('network');
    const training = useTraining();
    const draft = useRecipeDraft();
    const saveController = useSaveCurrentRun(draft.appliedVariation);
    const overlay = useDecisionOverlayPreference();
    const selection = useNetworkSelectionState();
    const destination = useLayoutStore((s) => s.destination);
    const workspaceTab = useLayoutStore((s) => s.workspaceTab);
    const setupTab = useLayoutStore((s) => s.setupTab);
    const resultsTab = useLayoutStore((s) => s.resultsTab);
    const inspectTab = useLayoutStore((s) => s.inspectTab);
    const audienceMode = useLayoutStore((s) => s.audienceMode);
    const activeLessonId = useLayoutStore((s) => s.activeLessonId);
    const cueDismissed = useLayoutStore((s) => s.lessonCueDismissed);
    const hasStartedLesson = useLayoutStore((s) => s.hasStartedLesson);
    const recordsReady = useExperimentMemoryStore((s) => s.hydrationStatus === 'ready');
    const hasRecords = useExperimentMemoryStore((s) => s.records.length > 0);
    const configError = useTrainingStore((s) => s.configError);
    const workerError = useTrainingStore((s) => s.workerError);
    const task = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe.task : null);
    const preparing = usePlaygroundStore((s) => s.preparation.status === 'preparing');
    const urlSyncNotice = usePlaygroundStore((s) => s.urlSyncNotice);
    const [utility, setUtility] = useState<UtilitySurface>(null);
    const [pendingNavigation, setPendingNavigation] = useState<(() => void) | null>(null);
    const [savedVisited, setSavedVisited] = useState(destination === 'saved-runs');
    const [saveFocusRequest, setSaveFocusRequest] = useState(0);
    const [lessonsVisited, setLessonsVisited] = useState(destination === 'lessons' || activeLessonId !== null);
    const [lessonHighlight, setLessonHighlight] = useState<LessonTarget | null>(null);
    const [exportTab, setExportTab] = useState<'setup' | 'code'>('setup');
    const exportRequest = useLayoutStore((state) => state.exportRequest);
    const backgroundRef = useRef<HTMLDivElement>(null);
    const focusExercise = useRef(false);
    const focusExerciseHeading = useCallback((heading: HTMLHeadingElement | null) => {
        if (heading && focusExercise.current) {
            focusExercise.current = false;
            heading.focus();
        }
    }, []);
    const draftRef = useRef(draft);
    draftRef.current = draft;
    const trainingRef = useRef(training);
    useEffect(() => { trainingRef.current = training; }, [training]);
    const stableReset = useCallback(() => trainingRef.current.reset(), []);
    const guard = (action: () => void) => {
        if (draft.dirty) { setUtility(null); setPendingNavigation(() => action); }
        else action();
    };
    const guardRef = useRef(guard);
    guardRef.current = guard;
    const requestGuardedNavigation = useCallback((action: () => void) => guardRef.current(action), []);
    useEffect(() => {
        if (!exportRequest) return;
        useLayoutStore.getState().clearExportRequest();
        requestGuardedNavigation(() => { setExportTab(exportRequest.mode); setUtility('exports'); });
    }, [exportRequest, requestGuardedNavigation]);
    const navigate = (next: Destination, tab?: WorkspaceTab) => {
        if (next === destination && (!tab || tab === workspaceTab)) return;
        const action = () => { setUtility(null); useLayoutStore.getState().navigate(next,tab); };
        if (destination === 'playground' && workspaceTab === 'setup') requestGuardedNavigation(action); else action();
    };
    const openSetup = useCallback((tab: SetupTab) => { setUtility(null); useLayoutStore.getState().openSetup(tab); }, []);
    const startExercise = () => {
        if (draft.commands.firstExercise()) {
            focusExercise.current = true;
            openSetup('dataset');
        }
    };
    const tryVariation = useCallback((record: ExperimentRunRecordV2) => {
        requestGuardedNavigation(() => {
            if (draftRef.current.commands.variation(record)) openSetup('training');
        });
    }, [openSetup, requestGuardedNavigation]);
    const openSaveRun = () => requestGuardedNavigation(() => {
        setUtility(null);
        useLayoutStore.getState().navigate('saved-runs');
        setSaveFocusRequest((request) => request + 1);
    });
    const openUtility = (surface: UtilitySurface) => {
        const action = () => setUtility(surface);
        if (surface === 'exports') guard(action); else action();
    };
    const shareSetup = () => guard(() => { setExportTab('setup'); setUtility('exports'); });
    useEffect(() => { if (destination === 'saved-runs') setSavedVisited(true); if (destination === 'lessons') setLessonsVisited(true); }, [destination]);

    const regression = task?.kind === 'regression';
    const resultTabs = regression ? REGRESSION_RESULTS_TABS : RESULTS_TABS;
    const effectiveResultsTab = regression && resultsTab === 'errors' ? 'boundary' : resultsTab;
    const stackedGraph = viewport.width < 1200 || activeLessonId !== null;
    const networkLayout = viewport.compact ? 'compact' : stackedGraph ? 'stacked' : 'wide';

    // Only mounted evidence demands expensive worker projections. This is the sole writer.
    useEffect(() => {
        const playground = destination === 'playground';
        const network = playground && workspaceTab === 'network';
        const inspect = playground && workspaceTab === 'inspect';
        const results = playground && workspaceTab === 'results';
        const next = { ...DEFAULT_DEMAND,
            needDecisionBoundary: (network && (!viewport.compact || networkRegion === 'prediction')) || (results && effectiveResultsTab !== 'learning'),
            needNeuronGrids: network && (!viewport.compact || networkRegion === 'network'),
            needLayerStats: inspect,
            needActivationHistograms: inspect && inspectTab === 'activations',
            needConfusionMatrix: results && effectiveResultsTab === 'errors',
        };
        const store = usePlaygroundStore.getState();
        if (Object.entries(next).some(([key,value]) => store.demand[key as keyof typeof next] !== value)) store.setDemand(next);
    }, [destination,workspaceTab,effectiveResultsTab,inspectTab,viewport.compact,networkRegion]);

    useEffect(() => {
        const handler = (event: KeyboardEvent) => {
            if (workerError || event.isComposing || document.querySelector('dialog[open], [role="menu"]')) return;
            const action = resolveTrainingShortcut(event);
            if (!action) return;
            const current = useTrainingStore.getState();
            const preparing = usePlaygroundStore.getState().preparation.status === 'preparing';
            const pausing = action === 'play-pause' && current.status === 'running';
            if (!pausing && (getTrainingUnavailableReason(current, preparing) || (action !== 'reset' && selectCheckpointBranchComplete(current)))) return;
            event.preventDefault();
            if (action === 'play-pause') { if (current.status === 'running') trainingRef.current.pause(); else trainingRef.current.play(); }
            else if (action === 'step') { if (current.status !== 'running') trainingRef.current.step(); }
            else trainingRef.current.reset();
        };
        window.addEventListener('keydown',handler);
        return () => window.removeEventListener('keydown',handler);
    }, [workerError]);

    useEffect(() => {
        if (pendingNavigation && !draft.dirty && !draft.submitted && !draft.error) {
            const action = pendingNavigation; setPendingNavigation(null); action();
        }
    }, [pendingNavigation, draft.dirty, draft.submitted, draft.error]);

    const dataset = task?.dataset ?? 'Experiment';
    const heading = destination === 'saved-runs' ? 'Saved runs' : destination === 'lessons' ? 'Learn by experimenting' : `${dataset.charAt(0).toUpperCase()}${dataset.slice(1)} experiment`;
    const description = destination === 'saved-runs' ? 'Keep the evidence. Compare what changed.' : destination === 'lessons' ? 'Small experiments. Ideas that become visible.' : 'Shape a network. Watch it learn.';
    return <div className="atelier" data-workspace={workspaceTab} data-destination={destination} data-short-viewport={viewport.short || undefined} ref={backgroundRef} data-lesson-highlight={lessonHighlight ?? undefined}>
        <a className="skip-link" href="#main-content" onClick={(event) => { event.preventDefault(); document.getElementById('main-content')?.focus(); }}>Skip to main content</a>
        <TrainingAnnouncements />
        <AtelierHeader destination={destination} onNavigate={(next) => navigate(next)} onUtility={openUtility} />
        <main data-guided={activeLessonId && destination === 'playground' ? 'true' : undefined} id="main-content" tabIndex={-1} className="atelier-main" aria-label="Neural network playground workspace">
            <div className="atelier-heading"><div><h1>{heading}</h1><p>{description}</p></div>{destination === 'playground' && <div className="atelier-heading-actions"><button type="button" onClick={shareSetup}>Share setup</button><button type="button" onClick={openSaveRun}>Save run…</button></div>}</div>
            {destination === 'playground' && urlSyncNotice && <div className="atelier-notice"><p role="status">{urlSyncNotice}</p><button type="button" onClick={() => usePlaygroundStore.getState().syncToUrl()}>Retry address update</button></div>}
            {configError && <div className="atelier-notice" role="alert"><p>{configError}</p><button type="button" onClick={() => useTrainingStore.getState().retryConfigSync()}>Retry configuration</button></div>}
            <AtelierTransport training={training} onCheckpoints={() => openUtility('checkpoints')} />
            <div className="atelier-workspace-layout">
            {destination === 'playground' && <div className="atelier-workspace">
                <Tabs panelPrefix="workspace" label="Experiment workspace" items={WORKSPACE_TABS} value={workspaceTab} onChange={(tab) => navigate('playground',tab)} />
                {workspaceTab === 'network' && recordsReady && !hasRecords && !cueDismissed && !hasStartedLesson && !activeLessonId && <aside className="atelier-notice atelier-invitation" aria-label="Getting started">
                    <p>Start small. Follow a guided experiment.</p>
                    {!draft.isFirstExercise && <button type="button" onClick={startExercise} disabled={preparing || draft.busy || draft.dirty || !draft.recipe}>{FIRST_RUN_EXERCISE.title}</button>}
                    <button type="button" onClick={() => navigate('lessons')}>Explore lessons</button>
                    <button type="button" aria-label="Dismiss lesson suggestion" onClick={() => useLayoutStore.getState().dismissLessonCue()}>Not now</button>
                </aside>}
                <TabPanels panelPrefix="workspace" items={WORKSPACE_TABS} value={workspaceTab} className="atelier-tab-content">
                    <ErrorBoundary title="Workspace unavailable" description="This view could not render. Your experiment remains available in the other views.">
                    {workspaceTab === 'setup' && <>
                        {draft.isFirstExercise && <section className="atelier-section" aria-labelledby="first-exercise-title">
                            <h2 id="first-exercise-title" ref={focusExerciseHeading} tabIndex={-1}>{FIRST_RUN_EXERCISE.title}</h2>
                            {draft.dirty && <p>Choose Apply changes to start this exercise, or Cancel to keep your current setup.</p>}
                            <p>{FIRST_RUN_EXERCISE.instructions}</p>
                        </section>}
                        <SetupEditor controller={draft} tab={setupTab} onTabChange={(tab) => useLayoutStore.getState().setSetupTab(tab)} />
                    </>}
                    {workspaceTab === 'network' && <>
                        <NetworkWorkspace
                            layout={networkLayout}
                            region={networkRegion}
                            onRegionChange={setNetworkRegion}
                            overlay={overlay}
                            selection={selection}
                            onSetup={openSetup}
                            onResults={() => {
                                useLayoutStore.getState().setResultsTab('boundary');
                                navigate('playground', 'results');
                            }}
                        />
                        <EvidenceMetrics />
                    </>}
                    {workspaceTab === 'results' && <section className="atelier-evidence">
                        <Tabs
                            panelPrefix="results"
                            label="Results views"
                            items={resultTabs}
                            value={effectiveResultsTab}
                            onChange={(tab) => {
                                if (tab === 'errors') overlay.setOverlayMode('misclassification');
                                useLayoutStore.getState().setResultsTab(tab);
                            }}
                        />
                        <TabPanels panelPrefix="results" items={resultTabs} value={effectiveResultsTab} tabIndex={0}>
                            {effectiveResultsTab === 'boundary' && <>
                                <OptionalView title="Prediction unavailable">
                                    <div className="atelier-results-boundary"><PredictionWorkspace overlay={overlay} details /></div>
                                </OptionalView>
                                <EvidenceMetrics />
                            </>}
                            {effectiveResultsTab === 'learning' && <>
                                <OptionalView title="Learning chart unavailable"><LossChart /></OptionalView>
                                <EvidenceMetrics />
                                <OptionalView title="Training explanation unavailable"><TrainingExplanationPanel /></OptionalView>
                                <CurrentRunCard />
                            </>}
                            {effectiveResultsTab === 'errors' && <div className="atelier-results-errors">
                                <div className="atelier-error-map">
                                    <h2>Prediction field</h2>
                                    <OptionalView title="Prediction unavailable"><PredictionWorkspace overlay={overlay} details /></OptionalView>
                                </div>
                                <OptionalView title="Confusion matrix unavailable"><ConfusionMatrix /></OptionalView>
                            </div>}
                        </TabPanels>
                    </section>}
                    {workspaceTab === 'inspect' && <section className="atelier-evidence"><OptionalView title="Workspace unavailable"><InspectionPanel onPause={training.pause} /></OptionalView></section>}
                    </ErrorBoundary>
                </TabPanels>
            </div>}
            {(savedVisited || destination === 'saved-runs') && <div hidden={destination !== 'saved-runs'}><OptionalView title="Saved runs unavailable"><RunHistoryPanel saveController={saveController} saveFocusRequest={saveFocusRequest} onTryVariation={tryVariation} variationDisabled={draft.busy} /></OptionalView></div>}
            {(lessonsVisited || destination === 'lessons') && <div className="atelier-lesson-host" hidden={destination !== 'lessons' && !activeLessonId}><OptionalView title="Lessons unavailable"><GuidedLessonPanel onNavigate={requestGuardedNavigation} onReset={stableReset} onHighlightChange={setLessonHighlight} /></OptionalView></div>}
            </div>
        </main>
        <footer className="atelier-footer"><span>NN·FORGE · Experiments stay in your browser</span><span>Data → network → prediction</span></footer>
        {!workerError && !pendingNavigation && utility && <Dialog title={utility === 'exports' ? 'Export / import' : utility === 'checkpoints' ? 'Session checkpoints' : utility === 'preferences' ? 'Guidance' : 'Shortcuts & help'} onClose={() => setUtility(null)}>
            {utility === 'checkpoints' && <OptionalView title="Checkpoints unavailable"><CheckpointPanel training={training} /></OptionalView>}
            {utility === 'exports' && <>
                <Tabs<'setup' | 'code'> panelPrefix="export" label="Export type" items={EXPORT_TABS} value={exportTab} onChange={setExportTab} />
                <TabPanels panelPrefix="export" items={EXPORT_TABS} value={exportTab} tabIndex={0}>
                    <OptionalView title="Export view unavailable">{exportTab === 'setup' ? <ConfigPanel onReset={stableReset} /> : <CodeExportPanel />}</OptionalView>
                </TabPanels>
            </>}
            {utility === 'preferences' && <><p>Choose how much explanation appears alongside the controls. Every feature remains available.</p><label htmlFor="atelier-guidance">Explanation density</label><select id="atelier-guidance" value={audienceMode} onChange={(event) => useLayoutStore.getState().setAudienceMode(event.target.value as 'beginner'|'explore'|'lab')}><option value="beginner">More</option><option value="explore">Standard</option><option value="lab">Compact</option></select></>}
            {utility === 'help' && <div className="atelier-help"><p>Build a dataset and network in Setup, apply the recipe, then train and inspect the results. Training continues when you change views.</p><dl>{TRAINING_SHORTCUTS.map((shortcut) => <div key={shortcut.code}><dt><kbd>{shortcut.label}</kbd></dt><dd>{shortcut.description}</dd></div>)}</dl><p>Shortcuts are inactive while editing fields or using a menu or dialog. Step is available while paused.</p><p>Saved runs hold evaluated evidence. Applying a saved recipe starts a fresh model; session checkpoints restore parameters and optimizer state. Eligible checkpoints can branch into a new run with seeded replay.</p><OptionalView title="Concept library unavailable"><ConceptLibrary /></OptionalView></div>}
        </Dialog>}
        {!workerError && pendingNavigation && <Dialog title="Apply your setup changes?" onClose={() => { if (!draft.busy) setPendingNavigation(null); }} alert>
            <p>You have one shared draft across Dataset, Network, and Training. Apply it before leaving, discard it, or stay to keep editing.</p>
            {draft.error && <div role="alert" className="atelier-error"><p>{draft.error}</p>{draft.submitted && <button type="button" onClick={draft.commands.retry}>Retry configuration</button>}</div>}
            <div className="atelier-dialog-actions"><button type="button" disabled={draft.busy || draft.submitted} onClick={() => setPendingNavigation(null)}>Stay</button><button type="button" disabled={draft.busy || draft.submitted} onClick={() => { draft.commands.cancel(); const action = pendingNavigation; setPendingNavigation(null); action(); }}>Discard changes</button><button type="button" className="atelier-primary" disabled={!draft.canApply} onClick={() => { void draft.commands.apply(); }}>Apply changes</button></div>
        </Dialog>}
        {workerError && <Dialog title="Worker connection lost" persistent backgroundRef={backgroundRef} description={`${workerError} Refresh the page to restart the playground.`} onClose={() => { /* Persistent recovery requires reload. */ }} alert><p>{workerError}</p><p>Refresh the page to restart the playground.</p><button type="button" className="atelier-primary" onClick={() => window.location.reload()}>Refresh page</button></Dialog>}
    </div>;
}

interface NetworkWorkspaceProps extends Omit<AtelierNetworkProps, 'graph' | 'prediction'> {
    selection: NetworkSelectionState;
    overlay: DecisionOverlayPreference;
}

// Controller state stays with the shell while frame subscriptions follow the evidence views.
function NetworkWorkspace({ selection, overlay, ...view }: NetworkWorkspaceProps) {
    const graph = useNetworkSelectionModel(selection, view.layout !== 'compact' || view.region === 'network');
    return <AtelierNetwork {...view} graph={graph} prediction={<PredictionWorkspace overlay={overlay} />} />;
}

const PredictionWorkspace = memo(function PredictionWorkspace({ overlay, details = false }: { overlay: DecisionOverlayPreference; details?: boolean }) {
    const status = useTrainingStore(selectDatasetPlotStatus);
    if (status !== 'ready') return <DatasetPlotStatus className="decision-boundary" status={status} />;
    return <PredictionEvidence overlay={overlay} details={details} />;
});

function PredictionEvidence({ overlay, details }: { overlay: DecisionOverlayPreference; details: boolean }) {
    const boundary = useDecisionBoundaryController(overlay);
    return <AtelierPrediction boundary={boundary} details={details} />;
}

function TrainingAnnouncements() {
    const status = useTrainingStore((s) => s.status);
    const pauseReason = useTrainingStore((s) => s.pauseReason);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const workerError = useTrainingStore((s) => s.workerError);
    const evidenceGenerationId = useTrainingStore((s) => s.evidenceGenerationId);
    const trainedRecipe = useTrainingStore((s) => s.trainedRecipe);
    const trainedRecipeSource = useTrainingStore((s) => s.trainedRecipeSource);
    return <AccessibilityAnnouncer
        status={status}
        pauseReason={pauseReason}
        workerError={workerError}
        pendingConfigSource={pendingConfigSource}
        configError={configError}
        configErrorSource={configErrorSource}
        evidenceGenerationId={evidenceGenerationId}
        trainedRecipe={trainedRecipe}
        trainedRecipeSource={trainedRecipeSource}
    />;
}
