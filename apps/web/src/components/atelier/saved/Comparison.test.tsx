import { render, screen, within } from '@testing-library/react';
import { PREPARED_PRESETS, type ExperimentRunRecordV2, type PairedEvaluation } from '@nn-playground/shared';
import { expect, it } from 'vitest';
import { makeSavedRunRecord } from '../../../test/savedRunFixtures.ts';
import { SavedRunComparison } from './Comparison.tsx';

const gaussian = PREPARED_PRESETS.find((preset) => preset.id === 'single-neuron')!.prepared;
const circles = PREPARED_PRESETS.find((preset) => preset.id === 'circle-one-layer')!.prepared;
const baselineId = '00000000-0000-0000-0000-000000000002';

function withEvaluations(record: ExperimentRunRecordV2, steps: readonly (readonly [number, number])[]): ExperimentRunRecordV2 {
    const saved = record.snapshot.evaluation;
    const evaluationHistory = steps.map(([step, dataLoss], index): PairedEvaluation => ({
        ...saved,
        evaluationId: index + 1,
        trigger: index === steps.length - 1 ? 'save' : 'manual-step',
        model: { ...saved.model, revision: step, step },
        test: { ...saved.test, values: { dataLoss } },
    }));
    const evaluation = evaluationHistory.at(-1)!;
    return {
        ...record,
        snapshot: { ...record.snapshot, model: evaluation.model, evaluation, evaluationHistory },
    };
}

function compare(current: ExperimentRunRecordV2, baseline: ExperimentRunRecordV2) {
    render(<SavedRunComparison current={current} baseline={baseline}
        currentLabel="Variation" baselineLabel="Baseline" showRecipeChanges onBack={() => {}} />);
}

function valuesFor(rowName: string) {
    return within(screen.getByRole('row', { name: new RegExp(`^${rowName}`) }))
        .getAllByRole('cell').map((cell) => cell.textContent);
}

it('compares the latest full evaluations at the greatest shared step instead of unequal endpoints', () => {
    const current = withEvaluations(makeSavedRunRecord(gaussian), [[0, 0.9], [20, 0.3], [20, 0.2], [70, 0.6]]);
    const baseline = withEvaluations(makeSavedRunRecord(gaussian, baselineId), [[0, 0.5], [20, 0.1], [20, 0.5], [50, 0.1]]);
    compare(current, baseline);

    expect(screen.getByText('Comparing full evaluations at shared step 20.')).toBeInTheDocument();
    expect(screen.getByText('Variation has lower test data loss by 0.3000')).toBeInTheDocument();
    expect(valuesFor('Evaluated step')).toEqual(['20', '20']);
    expect(valuesFor('Test data loss')).toEqual(['0.2000', '0.5000']);
});

it('names both saved endpoint steps when the only shared step comes from batch trends', () => {
    const current = withEvaluations(makeSavedRunRecord(gaussian), [[15, 0.2], [35, 0.4]]);
    const baseline = withEvaluations(makeSavedRunRecord(gaussian, baselineId), [[20, 0.5], [40, 0.6]]);
    const endpoint = current.snapshot.evaluation;
    compare({
        ...current,
        snapshot: {
            ...current.snapshot,
            trendHistory: [{
                model: { ...endpoint.model, revision: 20, step: 20 },
                dataset: endpoint.dataset,
                objectiveKey: endpoint.objectiveKey,
                basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: 20 },
                dataLoss: 0.3,
            }],
        },
    }, baseline);

    expect(screen.getByText('No shared full-evaluation step. Variation was saved at step 35 and Baseline was saved at step 40.')).toBeInTheDocument();
    expect(screen.queryByText(/lower test data loss|equal test data loss/i)).toBeNull();
    expect(valuesFor('Evaluated step')).toEqual(['35', '40']);
});

it('keeps the dataset and objective compatibility guard at a shared full-evaluation step', () => {
    const current = withEvaluations(makeSavedRunRecord(gaussian), [[10, 0.2], [35, 0.4]]);
    const baseline = withEvaluations(makeSavedRunRecord(circles, baselineId), [[10, 0.5], [40, 0.6]]);
    compare(current, baseline);

    expect(screen.getByText('Not directly comparable')).toBeInTheDocument();
    expect(screen.getByText('Dataset and objective identities must both match before losses can be ranked.')).toBeInTheDocument();
    expect(screen.queryByText(/lower test data loss|equal test data loss/i)).toBeNull();
});
