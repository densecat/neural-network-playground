import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { useState } from 'react';
import { ALL_FEATURES } from '@nn-playground/engine';
import { DEFAULT_EXPERIMENT_DOCUMENT, PREPARED_PRESETS } from '@nn-playground/shared';
import { useRecipeDraft, type SetupTab } from '../../../hooks/useRecipeDraft.ts';
import { usePlaygroundStore } from '../../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../../store/useTrainingStore.ts';
import * as datasetPreviewModel from '../../controls/datasetPreviewModel.ts';
import { SetupEditor } from './SetupEditor.tsx';
function Harness({ initialTab = 'dataset' }: { initialTab?: SetupTab }) {
    const controller = useRecipeDraft();
    const [tab, onTabChange] = useState(initialTab);
    return <SetupEditor {...{ controller, tab, onTabChange }} />;
}
beforeEach(async () => {await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);useTrainingStore.getState().finishConfigChange();useTrainingStore.setState({configError:null,configErrorSource:null});});
it('retains one draft across tabs and cancels every pending section', () => {
    render(<Harness/>);
    fireEvent.change(screen.getByLabelText('Samples'),{target:{value:'100'}});
    fireEvent.click(screen.getByRole('button',{name:'Training'}));
    fireEvent.change(screen.getByLabelText('Batch size'),{target:{value:'200'}});
    expect(screen.getByRole('button',{name:'Apply changes'})).toBeDisabled();
    fireEvent.click(screen.getByRole('button',{name:'Data'}));
    expect(screen.getByLabelText('Samples')).toHaveValue('100');
    fireEvent.change(screen.getByLabelText('Samples'),{target:{value:'1000'}});
    expect(screen.getByRole('button',{name:'Apply changes'})).toBeEnabled();
    fireEvent.click(screen.getByRole('button',{name:'Cancel'}));
    expect(screen.getByLabelText('Samples')).toHaveValue(String(DEFAULT_EXPERIMENT_DOCUMENT.recipe.data.sampleCount));
    fireEvent.click(screen.getByRole('button',{name:'Training'}));
    expect(screen.getByLabelText('Batch size')).toHaveValue(String(DEFAULT_EXPERIMENT_DOCUMENT.recipe.training.batchSize));
});

it('keeps thumbnails and the selected preview available at valid noise 75', () => {
    const { container } = render(<Harness/>);
    fireEvent.change(screen.getByLabelText('Noise (%)'), { target: { value: '75' } });
    expect(screen.getByRole('button', { name: 'Apply changes' })).toBeEnabled();
    expect(screen.queryByText('Preview unavailable')).not.toBeInTheDocument();
    expect(container.querySelectorAll('canvas')).toHaveLength(12);
});

it('generates the selected preview only when Data is visible and its inputs change', () => {
    const preview = vi.spyOn(datasetPreviewModel, 'deriveDatasetPreviewModel');
    const selectedCalls = () => preview.mock.calls.filter(([args]) => args.sampleCount !== undefined);
    const { data, task } = DEFAULT_EXPERIMENT_DOCUMENT.recipe;
    try {
        render(<Harness initialTab="training" />);
        fireEvent.change(screen.getByLabelText('Learning rate'), { target: { value: '0.02' } });
        expect(selectedCalls()).toEqual([]);

        fireEvent.click(screen.getByRole('button', { name: 'Data' }));
        expect(selectedCalls()).toEqual([[
            { datasetId: task.dataset, seed: data.seed, noise: data.noise, sampleCount: data.sampleCount },
        ]]);
        fireEvent.change(screen.getByLabelText('Train fraction (0.1–0.9)'), { target: { value: '0.8' } });
        expect(selectedCalls()).toHaveLength(1);

        fireEvent.change(screen.getByLabelText('Samples'), { target: { value: '100' } });
        fireEvent.click(screen.getByRole('button', { name: 'XOR' }));
        fireEvent.click(screen.getByRole('button', { name: 'Reshuffle seed' }));
        fireEvent.change(screen.getByLabelText('Noise (%)'), { target: { value: '75' } });
        expect(selectedCalls()).toHaveLength(5);
        expect(selectedCalls().at(-1)).toEqual([
            { datasetId: 'xor', seed: data.seed + 1, noise: 75, sampleCount: 100 },
        ]);
    } finally {
        preview.mockRestore();
    }
});

it('offers the complete canonical preset catalog as a cancellable local draft', () => {
    render(<Harness />);
    fireEvent.click(screen.getByText('Start from a preset'));
    for (const preset of PREPARED_PRESETS) {
        fireEvent.click(screen.getByRole('button', { name: `${preset.title} ${preset.description}` }));
        expect(screen.getByLabelText('Samples')).toHaveValue(String(preset.prepared.document.recipe.data.sampleCount));
        expect(screen.getByRole('button', { name: `${preset.title} ${preset.description}` })).toHaveAttribute('aria-pressed', 'true');
    }
    expect(usePlaygroundStore.getState().access).toMatchObject({ prepared: { document: DEFAULT_EXPERIMENT_DOCUMENT } });
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.getByRole('button', { name: 'Apply changes' })).toBeDisabled();
});

it('keeps all dataset contracts and all nine features available without publishing edits', () => {
    render(<Harness />);
    for (const [label, kind] of [['Three-Class', 'Multiclass classification'], ['Plane', 'Regression'], ['Circle', 'Binary classification']] as const) {
        fireEvent.click(screen.getByRole('button', { name: label }));
        expect(screen.getByText(`${kind} · draft preview`)).toBeInTheDocument();
    }
    for (const label of ['Circle', 'XOR', 'Gaussian', 'Spiral', 'Moons', 'Checker', 'Rings', 'Heart', 'Three-Class', 'Plane', 'Multi-Gauss']) {
        expect(screen.getByRole('button', { name: label })).toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole('button', { name: 'Inputs & layers' }));
    expect(screen.getAllByRole('checkbox')).toHaveLength(9);
    for (const feature of ALL_FEATURES) {
        const control = screen.getByRole('checkbox', { name: feature.label });
        if (!(control as HTMLInputElement).checked) fireEvent.click(control);
        expect(control).toBeChecked();
    }
    expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
});

it('keeps exact invalid neuron widths visible and links canonical field errors', () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Inputs & layers' }));
    fireEvent.change(screen.getByLabelText('Layer 1 neurons'), { target: { value: '17' } });
    const neurons = screen.getByLabelText(/^Layer 1 neurons/);
    expect(neurons).toHaveValue('17');
    expect(neurons).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('button', { name: 'Apply changes' })).toBeDisabled();
    fireEvent.change(neurons, { target: { value: '16' } });
    expect(screen.getByRole('button', { name: 'Apply changes' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Data' }));
    fireEvent.change(screen.getByLabelText('Noise (%)'), { target: { value: '101' } });
    const noise = screen.getByRole('textbox', { name: /^Noise/ });
    expect(noise).toHaveAttribute('aria-invalid', 'true');
    const errorId = noise.getAttribute('aria-describedby');
    expect(errorId).toBeTruthy();
    expect(document.getElementById(errorId!)).toHaveTextContent(/100/);
    fireEvent.change(noise, { target: { value: '' } });
    expect(noise).toHaveValue('');
});

it('supports zero through six layers and displays readable activation and initialization labels', () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Inputs & layers' }));
    while (screen.queryByRole('button', { name: 'Remove layer 1' })) {
        fireEvent.click(screen.getByRole('button', { name: 'Remove layer 1' }));
    }
    expect(screen.queryByLabelText(/Layer \d neurons/)).not.toBeInTheDocument();
    for (let index = 0; index < 6; index++) fireEvent.click(screen.getByRole('button', { name: '+ Add hidden layer' }));
    expect(screen.getByRole('button', { name: '+ Add hidden layer' })).toBeDisabled();
    for (const [value, label] of [['leakyRelu', 'Leaky ReLU'], ['swish', 'Swish']]) {
        fireEvent.change(screen.getByLabelText('Hidden activation'), { target: { value } });
        expect(screen.getByRole('option', { name: label })).toBeInTheDocument();
        expect(screen.getByLabelText('Hidden activation')).toHaveValue(value);
    }
    expect(screen.getByRole('option', { name: 'Uniform random' })).toHaveValue('uniform');
});

it('exposes optimizer, schedule, penalty, clipping and regression loss parameters', () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Plane' }));
    fireEvent.click(screen.getByRole('button', { name: 'Training' }));
    expect(screen.getByLabelText('Optimizer')).toHaveValue('sgd');
    expect(screen.getByLabelText('Schedule')).toHaveValue('constant');
    expect(screen.getByLabelText('Penalty')).toHaveValue('none');
    expect(screen.getByLabelText('Clipping')).toHaveValue('none');
    for (const label of ['Momentum', 'Beta 1', 'Beta 2', 'Epsilon', 'Interval (steps)', 'Gamma', 'Duration (steps)', 'Minimum rate', 'Coefficient', 'Maximum norm', 'Huber delta']) {
        expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
    }
    fireEvent.change(screen.getByLabelText('Optimizer'), { target: { value: 'adam' } });
    expect(screen.getByLabelText('Epsilon')).toHaveValue('1e-8');
    fireEvent.change(screen.getByLabelText('Optimizer'), { target: { value: 'sgd-momentum' } });
    expect(screen.queryByLabelText('Epsilon')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Momentum')).toHaveValue('0.9');
    fireEvent.change(screen.getByLabelText('Schedule'), { target: { value: 'step' } });
    expect(screen.getByLabelText('Interval (steps)')).toHaveValue('100');
    fireEvent.change(screen.getByLabelText('Schedule'), { target: { value: 'cosine' } });
    expect(screen.queryByLabelText('Gamma')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Duration (steps)')).toHaveValue('1000');
    fireEvent.change(screen.getByLabelText('Data loss'), { target: { value: 'huber' } });
    fireEvent.change(screen.getByLabelText('Huber delta'), { target: { value: '0.123456789' } });
    expect(screen.getByLabelText('Huber delta')).toHaveValue('0.123456789');
    fireEvent.change(screen.getByLabelText('Penalty'), { target: { value: 'l2' } });
    expect(screen.getByLabelText('Coefficient')).toHaveValue('0.001');
    fireEvent.change(screen.getByLabelText('Clipping'), { target: { value: 'global-norm' } });
    expect(screen.getByLabelText('Maximum norm')).toHaveValue('1');
    expect(screen.getByRole('button', { name: 'Apply changes' })).toBeEnabled();
    expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
});

it.each([
    {
        name: 'step decay and momentum',
        schedule: { kind: 'step', interval: 17, gamma: 0.63 },
        optimizer: { kind: 'sgd-momentum', momentum: 0.81 },
        gradientClipping: { kind: 'global-norm', maximumNorm: 2.5, scope: 'total-objective-gradient' },
        fields: [['Interval (steps)', '17'], ['Gamma', '0.63'], ['Momentum', '0.81'], ['Maximum norm', '2.5']],
    },
    {
        name: 'cosine decay and Adam',
        schedule: { kind: 'cosine', totalSteps: 2500, minimumRate: 0.004 },
        optimizer: { kind: 'adam', beta1: 0.81, beta2: 0.97, epsilon: 1e-8 },
        gradientClipping: { kind: 'none' },
        fields: [['Duration (steps)', '2500'], ['Minimum rate', '0.004'], ['Beta 1', '0.81'], ['Beta 2', '0.97'], ['Epsilon', '1e-8']],
    },
] as const)('loads exact recipe parameters for $name', async ({ schedule, optimizer, gradientClipping, fields }) => {
    const base = PREPARED_PRESETS.find(({ id }) => id === 'regression-plane')!.prepared.document;
    const result = await usePlaygroundStore.getState().replaceDocument({
        ...base,
        recipe: {
            ...base.recipe,
            training: { ...base.recipe.training, schedule, optimizer, gradientClipping },
            objective: {
                ...base.recipe.objective,
                dataLoss: { kind: 'huber', delta: 1.75 },
                penalty: { kind: 'l2', coefficient: 0.012, applyTo: 'weights' },
            },
        },
    });
    expect(result.ok).toBe(true);
    useTrainingStore.getState().finishConfigChange();

    render(<Harness initialTab="training" />);
    expect(screen.getByLabelText('Schedule')).toHaveValue(schedule.kind);
    expect(screen.getByLabelText('Optimizer')).toHaveValue(optimizer.kind);
    expect(screen.getByLabelText('Clipping')).toHaveValue(gradientClipping.kind);
    expect(screen.getByLabelText('Huber delta')).toHaveValue('1.75');
    expect(screen.getByLabelText('Coefficient')).toHaveValue('0.012');
    expect(screen.getByLabelText('Penalty')).toHaveDisplayValue('L2 (squared weights)');
    for (const [label, value] of fields) expect(screen.getByLabelText(label)).toHaveValue(value);
});

it('steps neuron counts within the supported limits without applying the draft', () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Inputs & layers' }));
    if (!screen.queryByLabelText('Layer 1 neurons')) fireEvent.click(screen.getByRole('button', { name: '+ Add hidden layer' }));
    const field = screen.getByLabelText('Layer 1 neurons');
    fireEvent.change(field, { target: { value: '1' } });
    expect(screen.getByRole('button', { name: 'Decrease layer 1 neurons' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Increase layer 1 neurons' }));
    expect(field).toHaveValue('2');
    fireEvent.change(field, { target: { value: '16' } });
    expect(screen.getByRole('button', { name: 'Increase layer 1 neurons' })).toBeDisabled();
    expect(usePlaygroundStore.getState().access).toMatchObject({ prepared: { document: DEFAULT_EXPERIMENT_DOCUMENT } });
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.getByRole('button', { name: 'Apply changes' })).toBeDisabled();
});
