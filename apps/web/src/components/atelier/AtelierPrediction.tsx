import type { DatasetStatus, TrainingStore } from '../../store/useTrainingStore.ts';
import { BoundaryEvidencePanel } from '../visualization/BoundaryEvidencePanel.tsx';
import { DecisionBoundaryCanvas } from '../visualization/DecisionBoundaryCanvas.tsx';
import type { DecisionBoundaryController } from '../visualization/useDecisionBoundaryController.ts';

export function selectDatasetPlotStatus({ datasetStatus, runtimeSyncPending }: Pick<TrainingStore, 'datasetStatus' | 'runtimeSyncPending'>): DatasetStatus {
    return datasetStatus === 'ready' && runtimeSyncPending ? 'loading' : datasetStatus;
}

export function DatasetPlotStatus({ className, status }: { className: string; status: Exclude<DatasetStatus, 'ready'> }) {
    return <div className={`${className} atelier-plot-state`}>
        {status === 'error'
            ? <p role="alert">The dataset could not be loaded.</p>
            : <p role="status">Loading dataset…</p>}
    </div>;
}

export function AtelierPrediction({ boundary, details }: { boundary: DecisionBoundaryController; details: boolean }) {
    const { model } = boundary;
    return <>
        <DecisionBoundaryCanvas model={model} />
        {details
            ? <BoundaryEvidencePanel controller={boundary} />
            : <p className="atelier-plot-caption">
                {model.kind !== 'empty' && model.kind !== 'unavailable' && model.provenance && <span className="atelier-artifact-step">Grid at step {model.provenance.model.step}.</span>}
                The model’s output across the input space. Select Results for evaluation and overlays.
            </p>}
    </>;
}
