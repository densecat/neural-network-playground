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
    const gridStep = model.kind !== 'empty' && model.kind !== 'unavailable' ? model.provenance?.model.step : undefined;
    return <>
        <DecisionBoundaryCanvas model={model} />
        {details
            ? <BoundaryEvidencePanel controller={boundary} />
            : <p className="atelier-plot-caption">
                <span className="atelier-artifact-step">{gridStep === undefined ? null : `Grid at step ${gridStep}.`}</span>
                The model’s output across the input space. Select Results for evaluation and overlays.
            </p>}
    </>;
}
