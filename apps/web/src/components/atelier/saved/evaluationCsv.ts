import type { ExperimentRunRecordV2 } from '@nn-playground/shared';

const HEADERS = [
    'series', 'step', 'revision', 'train_sample_count', 'test_sample_count',
    'train_data_loss', 'test_data_loss', 'train_total_objective',
    'batch_sample_count', 'batch_ema_data_loss',
];

export function savedEvaluationCsv(snapshot: ExperimentRunRecordV2['snapshot']): string {
    const batchRows = snapshot.trendHistory.map((point) => ({
        revision: point.model.revision,
        cells: [
            'mini-batch-ema', point.model.step, point.model.revision, '', '', '', '', '',
            point.basis.latestBatchSize, point.dataLoss,
        ],
    }));
    const evaluationRows = snapshot.evaluationHistory.map((point) => ({
        revision: point.model.revision,
        cells: [
            'full-evaluation', point.model.step, point.model.revision,
            point.train.basis.sampleCount, point.test.basis.sampleCount,
            point.train.values.dataLoss, point.test.values.dataLoss,
            point.objective.trainTotalObjective, '', '',
        ],
    }));
    const rows = [...batchRows, ...evaluationRows].sort((a, b) => a.revision - b.revision);
    // Cells are fixed series labels, numeric values, or blanks.
    return [HEADERS, ...rows.map(({ cells }) => cells)].map((row) => row.join(',')).join('\r\n') + '\r\n';
}
