import '../controls/exportUtilities.css';
import { useEffect, useRef, useState } from 'react';
import { isCheckpointBranchStepCount, MAX_CHECKPOINT_BRANCH_STEPS } from '@nn-playground/shared';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { selectScientificEvidence } from '../../store/evidenceSelectors.ts';

type CheckpointAction = 'restore' | 'branch';

export function CheckpointPanel({ training }: { training: TrainingHook }) {
    const timeline = useTrainingStore((s) => s.checkpointTimeline);
    const status = useTrainingStore((s) => s.status);
    const pending = useTrainingStore((s) => s.pendingConfigSource);
    const runtimeSyncPending = useTrainingStore((s) => s.runtimeSyncPending);
    const current = useTrainingStore((s) => selectScientificEvidence(s).currentModel?.step ?? 0);
    const [id, setId] = useState<number | null>(null);
    const applying = useRef(false);
    const [action, setAction] = useState<CheckpointAction | null>(null);
    const [budget, setBudget] = useState('50');
    const [error, setError] = useState<string | null>(null);
    const selected = timeline.checkpoints.find((point) => point.id === id) ?? timeline.checkpoints.at(-1);
    useEffect(() => {
        if (id !== null && !timeline.checkpoints.some((point) => point.id === id)) setId(null);
    }, [id, timeline.checkpoints]);
    if (!selected) return <p>No checkpoints yet. Start training to build a session timeline.</p>;
    const index = timeline.checkpoints.findIndex((point) => point.id === selected.id);
    const canBranch = selected.trajectoryGuarantee === 'same-recipe-training-trajectory';
    const trainingSteps = Number(budget);
    const validBudget = isCheckpointBranchStepCount(trainingSteps);
    const busy = action !== null;
    const actionsDisabled = status === 'running' || pending !== null || runtimeSyncPending || busy;
    const branch = timeline.branch;
    const branchSource = timeline.checkpoints.find((point) => point.id === branch?.sourceCheckpointId);
    const previous = timeline.checkpoints.find((point) => point.id === branch?.previousCheckpointId);
    const endpoint = timeline.checkpoints.find((point) => (
        point.id === timeline.liveCheckpointId && point.step === branch?.targetStep
    ));

    async function applyCheckpoint(nextAction: CheckpointAction, checkpointId: number) {
        const state = useTrainingStore.getState();
        if (applying.current || state.status === 'running' || state.pendingConfigSource !== null
            || state.runtimeSyncPending) return;
        applying.current = true;
        setAction(nextAction);
        setId(checkpointId);
        setError(null);
        try {
            if (nextAction === 'branch') await training.branchCheckpoint(checkpointId, trainingSteps);
            else await training.restoreCheckpoint(checkpointId);
        } catch (cause) {
            setError(cause instanceof Error ? cause.message.slice(0, 1000) : 'Checkpoint action failed.');
        } finally {
            applying.current = false;
            setAction(null);
        }
    }

    return <section className="atelier-checkpoints">
        <dl><div><dt>Current model</dt><dd>Step {current}</dd></div><div><dt>Selected checkpoint</dt><dd>{selected.label}</dd></div></dl>
        <label htmlFor="atelier-checkpoint-range">Checkpoint timeline</label>
        <input id="atelier-checkpoint-range" type="range" min={0} max={timeline.checkpoints.length - 1} step={1} value={index} disabled={busy} aria-valuetext={selected.label} onChange={(event) => setId(timeline.checkpoints[Number(event.target.value)].id)} />
        {timeline.restoredCheckpointId === selected.id && <p role="status">Restored checkpoint</p>}
        {branchSource && previous && <>
            <p role="status">Branched from {branchSource.label}. Previous live model saved at {previous.label}.</p>
            {branch?.targetStep !== undefined && <p>{current >= branch.targetStep
                ? `Branch completed at step ${branch.targetStep}.`
                : `Play to train through step ${branch.targetStep}.`}</p>}
            <button type="button" disabled={busy || selected.id === previous.id} onClick={() => setId(previous.id)}>
                Select previous live checkpoint
            </button>
            {endpoint && <button type="button" disabled={busy || selected.id === endpoint.id} onClick={() => setId(endpoint.id)}>
                Select branch endpoint
            </button>}
        </>}
        <p>Training loss {selected.trainDataLoss.toFixed(4)} · test loss {selected.testDataLoss.toFixed(4)}</p>
        <p>{canBranch
            ? 'With the same setup, training replays the same batches and learning rates. Branching keeps the previous live model as a checkpoint.'
            : 'This checkpoint preserves parameters and optimizer state. Future shuffles may differ.'}</p>
        <p>Checkpoints last for this session. Evaluation history and automatic-stop tracking restart when applying a checkpoint.</p>
        {canBranch && <div>
            <label htmlFor="atelier-checkpoint-budget">Optimizer steps for this branch</label>
            <input id="atelier-checkpoint-budget" type="number" min={1} max={MAX_CHECKPOINT_BRANCH_STEPS} step={1}
                value={budget} disabled={actionsDisabled} aria-invalid={!validBudget} onChange={(event) => setBudget(event.target.value)} />
        </div>}
        {error && <p role="alert" className="atelier-error">{error}</p>}
        <div className="atelier-dialog-actions">
            {status === 'running' && <button type="button" onClick={training.pause}>Pause training</button>}
            <button type="button" disabled={actionsDisabled} onClick={() => void applyCheckpoint('restore', selected.id)}>
                {action === 'restore' ? 'Restoring…' : `Restore ${selected.label}`}
            </button>
            <button type="button" className="atelier-primary" disabled={actionsDisabled || !canBranch || !validBudget} onClick={() => void applyCheckpoint('branch', selected.id)}>
                {action === 'branch' ? 'Branching…' : 'Branch from checkpoint'}
            </button>
        </div>
    </section>;
}
