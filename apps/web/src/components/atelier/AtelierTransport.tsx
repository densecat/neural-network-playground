import { useMemo } from 'react';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { useTrainingStore, type TrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { selectCheckpointBranchComplete, selectScientificEvidence } from '../../store/evidenceSelectors.ts';
import { getTrainingLifecycleUi } from '../controls/trainingLifecycle.ts';
import { Icon } from './ui.tsx';

export function getTrainingUnavailableReason(
    { datasetStatus, evidenceGenerationId, workerError, pendingConfigSource, runtimeSyncPending }: Pick<TrainingStore, 'datasetStatus' | 'evidenceGenerationId' | 'workerError' | 'pendingConfigSource' | 'runtimeSyncPending'>,
    preparing: boolean,
): string | null {
    if (datasetStatus === 'error' || workerError) return 'Training unavailable';
    if (preparing) return 'Preparing the experiment…';
    if (pendingConfigSource) return 'Applying configuration…';
    if (datasetStatus !== 'ready' || evidenceGenerationId === null) return 'Preparing the model and dataset…';
    if (runtimeSyncPending) return 'Updating the model…';
    return null;
}

export function AtelierTransport({ training, onCheckpoints }: { training: TrainingHook; onCheckpoints(): void }) {
    const status = useTrainingStore((s) => s.status);
    const pauseReason = useTrainingStore((s) => s.pauseReason);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const live = useTrainingStore((s) => s.latestLiveSignal);
    const evaluation = useTrainingStore((s) => s.latestEvaluation);
    const stepsPerFrame = useTrainingStore((s) => s.stepsPerFrame);
    const setStepsPerFrame = useTrainingStore((s) => s.setStepsPerFrame);
    const preparing = usePlaygroundStore((s) => s.preparation.status === 'preparing');
    const unavailableReason = useTrainingStore((s) => getTrainingUnavailableReason(s, preparing));
    const branchComplete = useTrainingStore(selectCheckpointBranchComplete);
    const model = useMemo(() => selectScientificEvidence({ latestLiveSignal: live, latestEvaluation: evaluation }).currentModel, [live, evaluation]);
    const lifecycle = getTrainingLifecycleUi({ status, pauseReason, pendingConfigSource });
    const running = status === 'running';
    const blocked = unavailableReason !== null;
    const branchFinished = branchComplete && pauseReason === 'max-steps';
    return <section className="atelier-transport" aria-label="Training controls" data-status={status} data-model-step={model?.step ?? 0} data-model-generation={model?.generationId} data-model-revision={model?.revision} data-lesson-target="transport">
        <div className="atelier-transport-row">
            <button type="button" className="atelier-primary" aria-label={running ? 'Pause training' : 'Play training'} disabled={!running && (blocked || branchComplete)} onClick={running ? training.pause : training.play}><Icon name={running ? 'pause' : 'play'} /><span>{running ? 'Pause' : 'Play'}</span></button>
            <button type="button" aria-label="Run one training step" disabled={running || blocked || branchComplete} onClick={training.step}><Icon name="step" /><span>Step</span></button>
            <button type="button" aria-label="Reset training" disabled={blocked} onClick={training.reset} title="Reset parameters, optimizer, and session evidence"><Icon name="reset" /><span>Reset</span></button>
            <label>Speed <select aria-label="Steps per frame" value={stepsPerFrame} onChange={(event) => setStepsPerFrame(Number(event.target.value))}>{[1,5,10,25,50].map((n) => <option key={n} value={n}>{n}×</option>)}</select></label>
            <div className="atelier-transport-position"><div><span>Step </span><strong>{model?.step.toLocaleString() ?? '—'}</strong></div><div><span>Epoch </span><strong>{model?.epoch ?? '—'}</strong></div><button type="button" onClick={onCheckpoints} aria-label="Checkpoints" title="Session checkpoints"><Icon name="history" /><span>Checkpoints</span></button></div>
        </div>
        <p className="atelier-transport-status" role="status" tabIndex={0}>{unavailableReason ?? (running ? 'Training · updates run locally in your browser' : branchFinished ? 'Branch finished · open Checkpoints to run another branch' : lifecycle.statusText ?? 'Ready · choose Play to begin')}</p>
    </section>;
}

export function EvidenceMetrics() {
    const live = useTrainingStore((s) => s.latestLiveSignal);
    const evaluation = useTrainingStore((s) => s.latestEvaluation);
    const evidence = useMemo(() => selectScientificEvidence({ latestLiveSignal: live, latestEvaluation: evaluation }), [live, evaluation]);
    const full = evidence.fullEvaluation;
    const metric = (value: number | undefined | null) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(4);
    return <div>
        <dl className="atelier-metrics">
            <div><dt>Training loss</dt><dd>{metric(full?.trainDataLoss)}<small>{full ? `${full.trainSampleCount} training samples` : 'Awaiting full evaluation'}</small></dd></div>
            <div><dt>Test loss</dt><dd>{metric(full?.testDataLoss)}<small>{full ? `${full.testSampleCount} held-out samples` : 'Awaiting full evaluation'}</small></dd></div>
            {full?.testAccuracy !== undefined ? <div><dt>Test accuracy</dt><dd>{(full.testAccuracy * 100).toFixed(1)}%<small>Full held-out evaluation</small></dd></div> : <div><dt>Training objective</dt><dd>{metric(full?.trainingObjective)}<small>Data loss + regularization</small></dd></div>}
            <div><dt>Batch loss · EMA</dt><dd>{metric(evidence.batchTrend?.dataLoss)}<small>{evidence.batchTrend ? `Batch signal at step ${evidence.batchTrend.step}` : 'No batch signal yet'}</small></dd></div>
        </dl>
        {full && <p className="atelier-plot-caption">Full evaluation at step <span className="atelier-evaluation-step">{full.step}</span>.{evidence.evaluationAgeSteps ? ` Current model is ${evidence.evaluationAgeSteps} steps ahead; these metrics describe the evaluated model.` : ' Evaluation matches the current step.'}</p>}
    </div>;
}
