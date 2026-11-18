import { useEffect, useRef, useState } from 'react';
import { StoredLearningChart } from './StoredLearningChart.tsx';
import { RecipeChanges } from './RecipeChanges.tsx';
import type { ExperimentRunRecordV2, PairedEvaluation } from '@nn-playground/shared';

function latestSharedEvaluation(
    current: ExperimentRunRecordV2['snapshot'],
    baseline: ExperimentRunRecordV2['snapshot'],
): [PairedEvaluation, PairedEvaluation] | null {
    const baselineByStep = new Map(
        [...baseline.evaluationHistory, baseline.evaluation].map((point) => [point.model.step, point]),
    );
    let matched: [PairedEvaluation, PairedEvaluation] | null = null;
    for (const point of [...current.evaluationHistory, current.evaluation]) {
        const baselinePoint = baselineByStep.get(point.model.step);
        if (baselinePoint && (!matched || point.model.step >= matched[0].model.step)) {
            matched = [point, baselinePoint];
        }
    }
    return matched;
}

function primaryRows(record: ExperimentRunRecordV2, e: PairedEvaluation): Record<string, string> {
    const r = record.recipe;
    const describe = (value: unknown): string => typeof value === 'object' && value !== null
        ? Object.entries(value).map(([key, child]) => key === 'kind' ? String(child) : `${key.replace(/([a-z])([A-Z])/g, '$1 $2')} ${String(child)}`).join(' · ')
        : String(value);
    return {
        'Dataset': `${r.task.dataset} · ${r.task.kind}`,
        'Data seed / noise / split': `${r.data.seed} / ${r.data.noise}% / ${r.data.trainFraction * 100}% training`,
        'Evaluated populations': `${e.train.basis.sampleCount} training · ${e.test.basis.sampleCount} test`,
        'Architecture': `${r.inputs.featureIds.length} → ${r.model.hiddenLayers.length ? r.model.hiddenLayers.join(' → ') + ' → ' : ''}${r.task.kind === 'multiclass-classification' ? 3 : 1}`,
        'Input features': r.inputs.featureIds.join(', '),
        'Activation / initialization': `${r.model.hiddenActivation} / ${r.model.initialization}`,
        'Model seed': String(r.model.seed),
        'Optimizer': describe(r.training.optimizer),
        'Learning rate / batch size': `${r.training.learningRate} / ${r.training.batchSize}`,
        'Schedule': describe(r.training.schedule),
        'Data loss': describe(r.objective.dataLoss),
        'Regularization': describe(r.objective.penalty),
        'Gradient clipping': describe(r.training.gradientClipping),
        'Evaluated step': String(e.model.step),
        'Train data loss': String(e.train.values.dataLoss),
        'Test data loss': String(e.test.values.dataLoss),
        'Training objective': String(e.objective.trainTotalObjective),
        'Test accuracy': e.test.values.accuracy === undefined ? 'Not applicable' : `${e.test.values.accuracy * 100}%`,
    };
}

function displayValue(key: string, value: string) {
    if (['Train data loss', 'Test data loss', 'Training objective'].includes(key)) return Number(value).toFixed(4);
    if (key === 'Test accuracy' && value.endsWith('%')) return `${parseFloat(value).toFixed(1)}%`;
    return value;
}

export function SavedRunComparison({ current, baseline, currentLabel, baselineLabel, showRecipeChanges, onBack }: {
    current: ExperimentRunRecordV2; baseline: ExperimentRunRecordV2;
    currentLabel: string; baselineLabel: string; showRecipeChanges?: boolean; onBack(): void;
}) {
    const heading = useRef<HTMLHeadingElement>(null);
    useEffect(() => { heading.current?.focus(); }, []);
    const [differences, setDifferences] = useState(false);
    const currentEndpoint = current.snapshot.evaluation;
    const baselineEndpoint = baseline.snapshot.evaluation;
    const comparable = currentEndpoint.dataset.datasetKey === baselineEndpoint.dataset.datasetKey
        && currentEndpoint.objectiveKey === baselineEndpoint.objectiveKey;
    const matched = showRecipeChanges ? latestSharedEvaluation(current.snapshot, baseline.snapshot) : null;
    const [left, right] = matched ?? [currentEndpoint, baselineEndpoint];
    const delta = left.test.values.dataLoss - right.test.values.dataLoss;
    const x = primaryRows(current, left);
    const y = primaryRows(baseline, right);
    const keys = Object.keys(x).filter((key) => !differences || x[key] !== y[key]);
    const records = [current,baseline];
    const maximumStep = Math.max(1,...records.flatMap((r) => [...r.snapshot.trendHistory.map((p) => p.model.step),...r.snapshot.evaluationHistory.map((p) => p.model.step)]));
    const maximumLoss = Math.max(.000001,...records.flatMap((r) => [...r.snapshot.trendHistory.map((p) => p.dataLoss),...r.snapshot.evaluationHistory.flatMap((p) => [p.train.values.dataLoss,p.test.values.dataLoss,p.objective.trainTotalObjective])]))*1.05;
    return <section role="group" aria-label={`Saved run comparison: ${currentLabel} and ${baselineLabel}`} className="saved-comparison">
        <button type="button" onClick={onBack}>← Saved runs</button>
        <h2 ref={heading} tabIndex={-1}>Compare experiments</h2>
        {showRecipeChanges && (matched
            ? <p>Comparing full evaluations at shared step {left.model.step.toLocaleString()}.</p>
            : <p>No shared full-evaluation step. {currentLabel} was saved at step {currentEndpoint.model.step.toLocaleString()} and {baselineLabel} was saved at step {baselineEndpoint.model.step.toLocaleString()}.</p>)}
        {!comparable && <>
            <p>Not directly comparable</p>
            <p>Dataset and objective identities must both match before losses can be ranked.</p>
        </>}
        {comparable && (!showRecipeChanges || matched) && <p>{delta === 0
            ? `Equal test data loss at ${left.test.values.dataLoss.toFixed(4)}`
            : `${delta < 0 ? currentLabel : baselineLabel} has lower test data loss by ${Math.abs(delta).toFixed(4)}`}</p>}
        {showRecipeChanges && <section aria-label="Recipe changes">
            <h3>Recipe changes</h3>
            <RecipeChanges baseline={baseline.recipe} variation={current.recipe} />
        </section>}
        <h3>Stored learning history</h3><div className="saved-learning-pair"><StoredLearningChart record={current} label={currentLabel} maximumStep={maximumStep} maximumLoss={maximumLoss} /><StoredLearningChart record={baseline} label={baselineLabel} maximumStep={maximumStep} maximumLoss={maximumLoss} /></div>
        <label className="saved-differences"><input type="checkbox" checked={differences} onChange={(event) => setDifferences(event.target.checked)} />Only show differences</label>
        <div className="saved-table-scroll" role="region" aria-label="Configuration and evidence" tabIndex={0}><table className="saved-comparison-table"><thead><tr><th scope="col">Configuration and evidence</th><th scope="col">{currentLabel}</th><th scope="col">{baselineLabel}</th></tr></thead><tbody>
            {keys.map((key) => <tr key={key} className={x[key] !== y[key] ? 'saved-different' : undefined}><th scope="row">{key}</th><td title={x[key]}>{displayValue(key, x[key])}</td><td title={y[key]}>{displayValue(key, y[key])}</td></tr>)}
            {!keys.length && <tr><td colSpan={3}>No differences in these configuration and evidence rows.</td></tr>}
        </tbody></table></div>
        <details className="saved-evidence-details"><summary>Exact identities and complete scientific evidence</summary>
            <p>All recorded values, recipe identities, model revisions and stored histories.</p>
            {records.map((record, index) => <section key={record.id}><h3>{index === 0 ? currentLabel : baselineLabel}</h3><pre>{JSON.stringify(record, null, 2)}</pre></section>)}
        </details>
        <p>Saved evidence contains no trained parameters. Applying a recipe initializes a fresh model.</p>
    </section>;
}
