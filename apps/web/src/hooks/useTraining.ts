// ── useTraining hook ──
// Manages the training loop, worker communication, and data synchronization.
// Phase 3: Uses useTrainingStore for runtime state, usePlaygroundStore for config.

import { useEffect, useRef, useCallback } from 'react';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import {
    useTrainingStore,
    type RejectedConfigRequest,
    type TrainedRecipeSource,
} from '../store/useTrainingStore.ts';
import { selectCheckpointBranchComplete, selectScientificEvidence } from '../store/evidenceSelectors.ts';
import {
    getWorkerApi,
    setupStreamChannel,
    postStreamCommand,
    startRenderLoop,
    stopRenderLoop,
    onSnapshot,
    newRunTo,
    discardPendingSnapshot,
    terminateWorker,
    emitE2EWorkerError,
} from '../worker/workerBridge.ts';
import {
    consumeE2EWorkerFault,
    E2E_WORKER_FAULT_MESSAGE,
} from '../testing/e2eFaults.ts';
import {
    getFrameBuffer,
    getFrameVersions,
    updateFrameBuffer,
    validateFrameBufferPatch,
    type FrameBufferPatch,
} from '../worker/frameBuffer.ts';
import {
    flattenBiases,
    flattenNeuronGrids,
    flattenWeights,
} from '../worker/frameBufferLayout.ts';
import type {
    DataPoint,
} from '@nn-playground/engine';
import { getDatasetContract } from '@nn-playground/engine';
import type {
    ArtifactProvenance,
    CheckpointTimeline,
    DatasetRevision,
    PairedEvaluation,
    PreparedExperimentDocumentV2,
    WorkerEvidenceMessageV2,
    WorkerArtifactProvenanceV2,
    WorkerExperimentRequestV2,
    WorkerProtocolErrorMessageV2,
    WorkerToMainMessage,
} from '@nn-playground/shared';
import {
    GRID_SIZE,
    WORKER_PROTOCOL_VERSION,
    isCheckpointBranchStepCount,
    parseArtifactProvenance,
    parseWorkerEvidenceMessageV2,
    parseCheckpointTimelineV2,
} from '@nn-playground/shared';
import type { WorkerDatasetPointsV2, WorkerExperimentResultV2 } from '../worker/training.worker.ts';

type TrainingWorkerRemote = Awaited<ReturnType<typeof getWorkerApi>>;
type InitializeExperimentResult = Awaited<
    ReturnType<TrainingWorkerRemote['initializeExperimentV2']>
>;

interface AcknowledgedRun {
    readonly runId: number;
    readonly owner: PreparedExperimentDocumentV2;
    readonly source: TrainedRecipeSource;
    readonly dataset: DatasetRevision;
    readonly evidence: WorkerEvidenceMessageV2;
    readonly framePatch: FrameBufferPatch;
    readonly timeline: CheckpointTimeline;
}

export interface TrainingHook {
    play: () => void;
    pause: () => void;
    step: () => Promise<void>;
    reset: () => Promise<void>;
    restoreCheckpoint: (id: number) => Promise<void>;
    branchCheckpoint: (id: number, trainingSteps?: number) => Promise<void>;
}

function getErrorMessage(error: unknown, fallback: string): string {
    if (error instanceof Error) return error.message;
    return typeof error === 'string' && error.length > 0 ? error : fallback;
}

function requirePreparedExperiment(): PreparedExperimentDocumentV2 {
    const access = usePlaygroundStore.getState().access;
    if (access.status === 'ready') return access.prepared;
    throw new Error(
        'Training is unavailable because the shared experiment URL is incompatible with version 2.',
    );
}

function currentPreparedExperiment(): PreparedExperimentDocumentV2 | null {
    const access = usePlaygroundStore.getState().access;
    return access.status === 'ready' ? access.prepared : null;
}

function isCurrentRunRequest(runRequestId: number, prepared: PreparedExperimentDocumentV2): boolean {
    const { access, runRequestId: currentRunRequestId } = usePlaygroundStore.getState();
    return currentRunRequestId === runRequestId
        && access.status === 'ready'
        && access.prepared.identities === prepared.identities;
}

function isCompletePointSplit(points: DataPoint[], count: number): boolean {
    if (!Array.isArray(points) || points.length !== count) return false;
    for (const point of points) {
        if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)
            || !Number.isFinite(point.label)) return false;
    }
    return true;
}

function validateDatasetPoints(
    response: WorkerDatasetPointsV2,
    runId: number,
    expected: DatasetRevision,
): void {
    const dataset = response?.dataset;
    if (response?.runId !== runId || dataset?.datasetKey !== expected.datasetKey
        || dataset?.generatorVersion !== expected.generatorVersion
        || dataset?.trainCount !== expected.trainCount || dataset?.testCount !== expected.testCount
        || !isCompletePointSplit(response.trainPoints, expected.trainCount)
        || !isCompletePointSplit(response.testPoints, expected.testCount)) {
        throw new Error('dataset points do not match the active experiment');
    }
}

export function createWorkerExperimentRequestV2(
    prepared: PreparedExperimentDocumentV2,
    requestId: number,
): WorkerExperimentRequestV2 {
    if (!Number.isSafeInteger(requestId) || requestId < 1) {
        throw new RangeError('worker requestId must be a positive safe integer');
    }
    return {
        type: 'initialize-experiment',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId,
        document: prepared.document,
        claimedIdentities: prepared.identities,
    };
}

/**
 * Direct manual-step RPCs can resolve before cadence evidence already queued
 * on the ordered MessagePort. Never jump the evaluation series over that gap.
 */
export function evidenceForDirectStep(
    evidence: WorkerEvidenceMessageV2,
    latestEvaluation: PairedEvaluation | null,
): WorkerEvidenceMessageV2 | null {
    const evaluation = evidence.latestEvaluation;
    if (!evaluation) return evidence;
    const latestId = latestEvaluation?.evaluationId ?? 0;
    if (evaluation.evaluationId === latestId
        || evaluation.evaluationId === latestId + 1) {
        return evidence;
    }
    if (!evidence.liveSignal && !evidence.artifacts) return null;
    return {
        type: 'evidence',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        ...(evidence.liveSignal ? { liveSignal: evidence.liveSignal } : {}),
        ...(evidence.artifacts ? { artifacts: evidence.artifacts } : {}),
    };
}

function getTotalNeuronCount(layerSizes: number[]): number {
    let total = 0;
    for (let i = 1; i < layerSizes.length; i++) {
        total += layerSizes[i];
    }
    return total;
}

function deepFreezeDirectArtifact<T>(value: T): Readonly<T> {
    if (typeof value !== 'object' || value === null || Object.isFrozen(value)) return value;
    for (const key of Reflect.ownKeys(value)) {
        deepFreezeDirectArtifact((value as Record<PropertyKey, unknown>)[key]);
    }
    return Object.freeze(value);
}

function directArtifact(
    result: WorkerExperimentResultV2,
    key: keyof WorkerArtifactProvenanceV2,
    expectedBasis: ArtifactProvenance['basis'],
): ArtifactProvenance {
    const raw = result.artifacts?.[key];
    if (raw === undefined) throw new Error(`direct V2 ${key} payload requires provenance`);
    const parsed = parseArtifactProvenance(raw);
    const evaluation = result.evidence.latestEvaluation;
    const sameModel = evaluation !== undefined
        && parsed.model.generationId === evaluation.model.generationId
        && parsed.model.revision === evaluation.model.revision
        && parsed.model.step === evaluation.model.step
        && parsed.model.epoch === evaluation.model.epoch;
    const sameBasis = (() => {
        if (parsed.basis.kind !== expectedBasis.kind) return false;
        switch (expectedBasis.kind) {
            case 'prediction-grid':
                return parsed.basis.kind === 'prediction-grid'
                    && parsed.basis.pointCount === expectedBasis.pointCount
                    && parsed.basis.domain.every(
                        (value, index) => value === expectedBasis.domain[index],
                    );
            case 'bounded-sample':
                return parsed.basis.kind === 'bounded-sample'
                    && parsed.basis.split === expectedBasis.split
                    && parsed.basis.sampleCount === expectedBasis.sampleCount
                    && parsed.basis.populationCount === expectedBasis.populationCount;
            case 'full-split':
                return parsed.basis.kind === 'full-split'
                    && parsed.basis.split === expectedBasis.split
                    && parsed.basis.sampleCount === expectedBasis.sampleCount
                    && parsed.basis.populationCount === expectedBasis.populationCount;
            case 'parameter-grid':
                return parsed.basis.kind === 'parameter-grid'
                    && parsed.basis.sampleCount === expectedBasis.sampleCount
                    && parsed.basis.parameterPositions === expectedBasis.parameterPositions;
        }
    })();
    if (parsed.model.generationId !== result.runId
        || evaluation === undefined
        || !sameModel
        || parsed.dataset.generatorVersion !== evaluation.dataset.generatorVersion
        || parsed.dataset.datasetKey !== evaluation.dataset.datasetKey
        || parsed.dataset.trainCount !== evaluation.dataset.trainCount
        || parsed.dataset.testCount !== evaluation.dataset.testCount
        || parsed.objectiveKey !== evaluation.objectiveKey
        || !sameBasis) {
        throw new Error(`direct V2 ${key} provenance identity or basis mismatch`);
    }
    return deepFreezeDirectArtifact(parsed) as ArtifactProvenance;
}

function validateEvidenceAgainstPrepared(
    value: unknown,
    prepared: PreparedExperimentDocumentV2,
): WorkerEvidenceMessageV2 {
    const evidence = parseWorkerEvidenceMessageV2(value);
    const sampleCount = prepared.compiled.data.sampleCount;
    const trainCount = sampleCount === 1
        ? 1
        : Math.min(sampleCount - 1, Math.max(
            1,
            Math.floor(sampleCount * prepared.compiled.data.trainFraction),
        ));
    const testCount = sampleCount - trainCount;
    const generatorVersion = getDatasetContract(prepared.compiled.data.dataset).generatorVersion;
    const identities = [
        evidence.liveSignal,
        evidence.latestEvaluation,
        ...Object.values(evidence.artifacts ?? {}),
    ].filter((entry): entry is NonNullable<typeof entry> => entry !== undefined);
    if (identities.some((entry) => (
        entry.dataset.generatorVersion !== generatorVersion
        || entry.dataset.datasetKey !== prepared.identities.datasetKey
        || entry.dataset.trainCount !== trainCount
        || entry.dataset.testCount !== testCount
        || entry.objectiveKey !== prepared.identities.objectiveKey
    ))) {
        throw new Error('worker evidence does not match the active prepared experiment');
    }
    const evaluation = evidence.latestEvaluation;
    if (evaluation !== undefined) {
        const validSide = (
            side: PairedEvaluation['train'] | PairedEvaluation['test'],
            count: number,
        ): boolean => {
            const { accuracy, confusionMatrix } = side.values;
            if (prepared.compiled.task.kind === 'regression') {
                return accuracy === undefined && confusionMatrix === undefined;
            }
            if (accuracy === undefined || accuracy < 0 || accuracy > 1
                || confusionMatrix === undefined) return false;
            if (prepared.compiled.task.kind === 'binary-classification') {
                return !('classCount' in confusionMatrix)
                    && confusionMatrix.tp + confusionMatrix.tn
                        + confusionMatrix.fp + confusionMatrix.fn === count
                    && accuracy === (confusionMatrix.tp + confusionMatrix.tn) / count;
            }
            return 'classCount' in confusionMatrix
                && confusionMatrix.counts.reduce((sum, value) => sum + value, 0) === count
                && accuracy === (
                    confusionMatrix.counts[0]
                    + confusionMatrix.counts[4]
                    + confusionMatrix.counts[8]
                ) / count;
        };
        if (!validSide(evaluation.train, trainCount)
            || !validSide(evaluation.test, testCount)) {
            throw new Error('worker evidence task metrics do not match the active prepared task');
        }
    }
    return evidence;
}

function preflightStrictV2Result(
    result: WorkerExperimentResultV2,
    expectedPrepared: PreparedExperimentDocumentV2,
    expectedTrigger: 'initial' | 'manual-step' | 'checkpoint' | 'restore',
): WorkerEvidenceMessageV2 {
    const { snapshot } = result;
    const evidence = validateEvidenceAgainstPrepared(result.evidence, expectedPrepared);
    const evaluation = evidence.latestEvaluation;
    if (result.identities?.canonicalRecipeKey
            !== expectedPrepared.identities.canonicalRecipeKey
        || result.identities?.recipeFingerprint
            !== expectedPrepared.identities.recipeFingerprint
        || result.identities?.datasetKey !== expectedPrepared.identities.datasetKey
        || result.identities?.objectiveKey !== expectedPrepared.identities.objectiveKey) {
        throw new Error('direct V2 result prepared identities do not match the active recipe');
    }
    const expectedTrainCount = Math.min(
        expectedPrepared.compiled.data.sampleCount - 1,
        Math.max(
            1,
            Math.floor(
                expectedPrepared.compiled.data.sampleCount
                * expectedPrepared.compiled.data.trainFraction,
            ),
        ),
    );
    const expectedTestCount = expectedPrepared.compiled.data.sampleCount - expectedTrainCount;
    if (evaluation === undefined
        || result.runId !== evaluation.model.generationId
        || snapshot.step !== evaluation.model.step
        || snapshot.epoch !== evaluation.model.epoch
        || (evidence.liveSignal !== undefined && (
            evidence.liveSignal.model.generationId !== result.runId
            || evidence.liveSignal.model.revision !== evaluation.model.revision
            || evidence.liveSignal.model.step !== evaluation.model.step
            || evidence.liveSignal.model.epoch !== evaluation.model.epoch
        ))
        || evaluation.dataset.datasetKey !== expectedPrepared.identities.datasetKey
        || evaluation.dataset.generatorVersion !== getDatasetContract(
            expectedPrepared.compiled.data.dataset,
        ).generatorVersion
        || evaluation.objectiveKey !== expectedPrepared.identities.objectiveKey
        || evaluation.dataset.trainCount !== expectedTrainCount
        || evaluation.dataset.testCount !== expectedTestCount) {
        throw new Error('direct V2 result requires an exact current evidence pair');
    }
    const checkpointTimeline = parseCheckpointTimelineV2(result.checkpointTimeline);
    if (expectedTrigger === 'initial' && (
        evaluation.evaluationId !== 1
        || evaluation.trigger !== 'initial'
        || evaluation.model.revision !== 0
        || evaluation.model.step !== 0
        || evaluation.model.epoch !== 0
        || evidence.liveSignal !== undefined
    )) {
        throw new Error('fresh direct V2 result requires the initial revision-zero evaluation');
    }
    if (expectedTrigger !== 'initial' && evaluation.trigger !== expectedTrigger) {
        throw new Error(`direct V2 result requires a ${expectedTrigger} evaluation`);
    }
    if (expectedTrigger === 'restore' && evidence.liveSignal !== undefined) {
        throw new Error('restore direct V2 result cannot retain a live training signal');
    }
    if (expectedTrigger === 'initial' && (
        checkpointTimeline.checkpoints.length !== 1
        || checkpointTimeline.checkpoints[0]?.step !== 0
        || checkpointTimeline.liveCheckpointId !== checkpointTimeline.checkpoints[0]?.id
        || checkpointTimeline.restoredCheckpointId !== null
        || checkpointTimeline.branch !== undefined
    )) {
        throw new Error('fresh direct V2 result requires one initial checkpoint summary');
    }
    const validateTaskSide = (
        side: PairedEvaluation['train'] | PairedEvaluation['test'],
        expectedCount: number,
    ): boolean => {
        const { accuracy, confusionMatrix } = side.values;
        if (expectedPrepared.compiled.task.kind === 'regression') {
            return accuracy === undefined && confusionMatrix === undefined;
        }
        if (accuracy === undefined || !Number.isFinite(accuracy) || accuracy < 0 || accuracy > 1
            || confusionMatrix === undefined) return false;
        if (expectedPrepared.compiled.task.kind === 'binary-classification') {
            return !('classCount' in confusionMatrix)
                && confusionMatrix.tp + confusionMatrix.tn
                    + confusionMatrix.fp + confusionMatrix.fn === expectedCount
                && accuracy === (confusionMatrix.tp + confusionMatrix.tn) / expectedCount;
        }
        return 'classCount' in confusionMatrix
            && confusionMatrix.classCount === 3
            && confusionMatrix.classLabels.every((label, index) => label === index)
            && confusionMatrix.counts.reduce((sum, count) => sum + count, 0) === expectedCount
            && accuracy === (
                confusionMatrix.counts[0]
                + confusionMatrix.counts[4]
                + confusionMatrix.counts[8]
            ) / expectedCount;
    };
    if (!validateTaskSide(evaluation.train, expectedTrainCount)
        || !validateTaskSide(evaluation.test, expectedTestCount)) {
        throw new Error('direct V2 result task metrics do not match the prepared task');
    }
    if (!Number.isSafeInteger(snapshot.step)
        || snapshot.step < 0
        || !Number.isSafeInteger(snapshot.epoch)
        || snapshot.epoch < 0
        || snapshot.gridSize !== GRID_SIZE
    ) {
        throw new Error('direct V2 result contains invalid snapshot scalars');
    }

    const layerSizes = [
        expectedPrepared.compiled.network.inputSize,
        ...expectedPrepared.compiled.network.hiddenLayers,
        expectedPrepared.compiled.network.outputSize,
    ];
    if (!Array.isArray(snapshot.weights)
        || snapshot.weights.length !== layerSizes.length - 1
        || !Array.isArray(snapshot.biases)
        || snapshot.biases.length !== layerSizes.length - 1) {
        throw new Error('direct V2 result parameter topology mismatch');
    }
    for (let layerIndex = 0; layerIndex < layerSizes.length - 1; layerIndex++) {
        const weights = snapshot.weights[layerIndex];
        const biases = snapshot.biases[layerIndex];
        if (!Array.isArray(weights)
            || weights.length !== layerSizes[layerIndex + 1]
            || !Array.isArray(biases)
            || biases.length !== layerSizes[layerIndex + 1]) {
            throw new Error('direct V2 result parameter topology mismatch');
        }
        for (const row of weights) {
            if (!Array.isArray(row)
                || row.length !== layerSizes[layerIndex]
                || row.some((value) => (
                    !Number.isFinite(value) || !Number.isFinite(Math.fround(value))
                ))) {
                throw new Error('direct V2 result contains malformed or non-finite weights');
            }
        }
        if (biases.some((value) => (
            !Number.isFinite(value) || !Number.isFinite(Math.fround(value))
        ))) {
            throw new Error('direct V2 result contains non-finite biases');
        }
    }

    const gridPointCount = snapshot.gridSize * snapshot.gridSize;
    const outputGrid = snapshot.outputGrid;
    if (!(outputGrid instanceof Float32Array) && !Array.isArray(outputGrid)) {
        throw new Error('direct V2 result output grid has an invalid representation');
    }
    if ((outputGrid.length !== 0 && outputGrid.length !== gridPointCount)
        || Array.from(outputGrid).some((value) => (
            !Number.isFinite(value) || !Number.isFinite(Math.fround(value))
        ))) {
        throw new Error('direct V2 result output grid has an invalid size or value');
    }
    if (snapshot.multiclassBoundary !== undefined) {
        const boundary = snapshot.multiclassBoundary;
        if (outputGrid.length > 0
            || boundary.gridSize !== snapshot.gridSize
            || !(boundary.classGrid instanceof Uint8Array)
            || !(boundary.confidenceGrid instanceof Float32Array)
            || boundary.classGrid.length !== gridPointCount
            || boundary.confidenceGrid.length !== gridPointCount
            || boundary.classGrid.some((value) => value > 2)
            || boundary.confidenceGrid.some((value) => (
                !Number.isFinite(value) || value < 0 || value > 1
            ))) {
            throw new Error('direct V2 result multiclass boundary is malformed');
        }
    }
    if (snapshot.neuronGrids !== undefined) {
        const expectedNeuronValues = layerSizes.slice(1).reduce((sum, size) => sum + size, 0)
            * gridPointCount;
        const validFlat = snapshot.neuronGrids instanceof Float32Array
            && snapshot.neuronGrids.length === expectedNeuronValues
            && !snapshot.neuronGrids.some((value) => !Number.isFinite(value));
        const validNested = Array.isArray(snapshot.neuronGrids)
            && snapshot.neuronGrids.length === expectedNeuronValues / gridPointCount
            && snapshot.neuronGrids.every((grid) => (
                (Array.isArray(grid) || grid instanceof Float32Array)
                && grid.length === gridPointCount
                && !Array.from(grid).some((value) => (
                    !Number.isFinite(value) || !Number.isFinite(Math.fround(value))
                ))
            ));
        if (!validFlat && !validNested) {
            throw new Error('direct V2 result neuron grids have an invalid size or value');
        }
    }
    if (snapshot.layerStats === undefined && result.layerStatsGradientRevision !== undefined) {
        throw new Error('direct V2 result has a surplus layer statistics revision');
    }
    if (snapshot.layerStats !== undefined && (
        snapshot.layerStats.length !== layerSizes.length - 1
        || snapshot.layerStats.some((stats) => (
            Object.keys(stats).sort().join(',')
                !== 'activationStd,meanAbsGradient,meanAbsWeight,meanActivation'
            || Object.values(stats).some((value) => !Number.isFinite(value))
            || stats.activationStd < 0
            || stats.meanAbsWeight < 0
            || stats.meanAbsGradient < 0
        ))
    )) {
        throw new Error('direct V2 result layer statistics are malformed');
    }
    if (snapshot.activationHistograms !== undefined) {
        const histogram = snapshot.activationHistograms;
        const histogramSampleCount = Math.min(128, expectedTrainCount);
        const validLayers = histogram.layers.length === layerSizes.length - 1
            && histogram.layers.every((layer, index) => (
                layer.layerIndex === index
                && layer.binCount === 12
                && Number.isFinite(layer.binStart)
                && Number.isFinite(layer.binWidth)
                && layer.binWidth > 0
                && Number.isFinite(layer.minActivation)
                && Number.isFinite(layer.maxActivation)
                && layer.minActivation <= layer.maxActivation
                && layer.totalCount === histogramSampleCount * layerSizes[index + 1]
                && Number.isSafeInteger(layer.nearZeroCount)
                && layer.nearZeroCount >= 0
                && layer.nearZeroCount <= layer.totalCount
                && Number.isSafeInteger(layer.saturatedCount)
                && layer.saturatedCount >= 0
                && layer.saturatedCount <= layer.totalCount
                && Array.from(
                    histogram.bins.subarray(index * 12, (index + 1) * 12),
                ).reduce((sum, count) => sum + count, 0) === layer.totalCount
            ));
        if (!(histogram.bins instanceof Float32Array)
            || histogram.bins.length !== histogram.layers.length * 12
            || histogram.bins.some((value) => (
                !Number.isSafeInteger(value) || value < 0
            ))
            || !validLayers) {
            throw new Error('direct V2 result activation histogram is malformed');
        }
    }
    return evidence;
}

function buildStrictV2FramePatch(
    result: WorkerExperimentResultV2,
    replaceAbsentArtifacts: boolean,
    expectedPrepared: PreparedExperimentDocumentV2,
    expectedTrigger: 'initial' | 'manual-step' | 'checkpoint' | 'restore' = (
        replaceAbsentArtifacts ? 'initial' : 'manual-step'
    ),
): FrameBufferPatch {
    const { snapshot } = result;
    const evidence = preflightStrictV2Result(result, expectedPrepared, expectedTrigger);
    const parameterModel = evidence.latestEvaluation!.model;
    const currentFrame = getFrameBuffer();
    const { buffer: weights, layerSizes } = flattenWeights(snapshot.weights);
    const biases = flattenBiases(snapshot.biases);
    const patch: Parameters<typeof updateFrameBuffer>[0] = {
        weights,
        biases,
        weightLayout: { layerSizes },
        parameterProvenance: {
            model: parameterModel,
            recipeFingerprint: result.identities.recipeFingerprint,
        },
    };

    const hasScalarBoundary = snapshot.outputGrid.length > 0;
    const hasMulticlassBoundary = snapshot.multiclassBoundary !== undefined;
    if (hasScalarBoundary && hasMulticlassBoundary) {
        throw new Error('direct V2 snapshot cannot contain scalar and multiclass boundaries');
    }
    const hasCurrentBoundary = currentFrame.outputGrid !== null
        || currentFrame.multiclassClassGrid !== null
        || currentFrame.multiclassConfidenceGrid !== null
        || currentFrame.multiclassBoundaryLayout !== null
        || currentFrame.decisionBoundaryProvenance !== null;
    const canReuseCurrentBoundary = (
        currentFrame.outputGrid !== null
        || currentFrame.multiclassClassGrid !== null
        || currentFrame.multiclassConfidenceGrid !== null
    ) && currentFrame.decisionBoundaryProvenance?.model.generationId === result.runId;
    const clearCurrentBoundary = !hasScalarBoundary
        && !hasMulticlassBoundary
        && (replaceAbsentArtifacts || (hasCurrentBoundary && !canReuseCurrentBoundary));
    if (hasScalarBoundary || hasMulticlassBoundary || clearCurrentBoundary) {
        patch.outputGrid = hasScalarBoundary
            ? snapshot.outputGrid instanceof Float32Array
                ? snapshot.outputGrid
                : new Float32Array(snapshot.outputGrid)
            : null;
        patch.gridSize = hasScalarBoundary || hasMulticlassBoundary ? snapshot.gridSize : 0;
        patch.multiclassClassGrid = snapshot.multiclassBoundary?.classGrid ?? null;
        patch.multiclassConfidenceGrid = snapshot.multiclassBoundary?.confidenceGrid ?? null;
        patch.multiclassBoundaryLayout = snapshot.multiclassBoundary === undefined
            ? null
            : {
                gridSize: snapshot.multiclassBoundary.gridSize,
                classCount: 3,
                classLabels: [0, 1, 2],
            };
        patch.decisionBoundaryProvenance = hasScalarBoundary || hasMulticlassBoundary
            ? directArtifact(result, 'decisionBoundary', {
                kind: 'prediction-grid',
                pointCount: snapshot.gridSize * snapshot.gridSize,
                domain: [-1, 1, -1, 1],
            })
            : null;
    }

    let neuronGrids: Float32Array | null = null;
    let neuronGridLayout: { count: number; gridSize: number } | null = null;
    if (snapshot.neuronGrids !== undefined && snapshot.neuronGrids.length > 0) {
        if (snapshot.neuronGrids instanceof Float32Array) {
            neuronGrids = snapshot.neuronGrids;
            neuronGridLayout = { count: getTotalNeuronCount(layerSizes), gridSize: snapshot.gridSize };
        } else {
            const flattened = flattenNeuronGrids(snapshot.neuronGrids, snapshot.gridSize);
            neuronGrids = flattened.buffer;
            neuronGridLayout = flattened.layout;
        }
    }
    const hasCurrentNeuronGrids = currentFrame.neuronGrids !== null
        || currentFrame.neuronGridLayout !== null
        || currentFrame.neuronGridsProvenance !== null;
    const canReuseCurrentNeuronGrids = currentFrame.neuronGrids !== null
        && currentFrame.neuronGridLayout !== null
        && currentFrame.neuronGridsProvenance?.model.generationId === result.runId;
    const clearCurrentNeuronGrids = neuronGrids === null
        && (replaceAbsentArtifacts || (hasCurrentNeuronGrids && !canReuseCurrentNeuronGrids));
    if (neuronGrids !== null || clearCurrentNeuronGrids) {
        patch.neuronGrids = neuronGrids;
        patch.neuronGridLayout = neuronGridLayout;
        patch.neuronGridsProvenance = neuronGrids === null
            ? null
            : directArtifact(result, 'neuronGrids', {
                kind: 'prediction-grid',
                pointCount: snapshot.gridSize * snapshot.gridSize,
                domain: [-1, 1, -1, 1],
            });
    }

    if (snapshot.layerStats !== undefined) {
        patch.layerStats = snapshot.layerStats;
        const populationCount = result.evidence.latestEvaluation!.dataset.trainCount;
        const provenance = directArtifact(result, 'activationStatistics', {
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(128, populationCount),
            populationCount,
        });
        const gradientRevision = result.layerStatsGradientRevision;
        if (!Number.isSafeInteger(gradientRevision)
            || (gradientRevision as number) < 0
            || (gradientRevision as number) > provenance.model.revision) {
            throw new Error('direct V2 layer statistics require a valid gradient revision');
        }
        patch.layerStatsProvenance = provenance;
        patch.layerStatsGradientRevision = gradientRevision!;
    } else {
        const hasCurrentLayerStats = currentFrame.layerStats !== null
            || currentFrame.layerStatsProvenance !== null
            || currentFrame.layerStatsGradientRevision !== null;
        const canReuseCurrentLayerStats = currentFrame.layerStats !== null
            && currentFrame.layerStatsProvenance?.model.generationId === result.runId
            && Number.isSafeInteger(currentFrame.layerStatsGradientRevision)
            && (currentFrame.layerStatsGradientRevision as number) >= 0;
        if (replaceAbsentArtifacts || (hasCurrentLayerStats && !canReuseCurrentLayerStats)) {
            patch.layerStats = null;
            patch.layerStatsProvenance = null;
            patch.layerStatsGradientRevision = null;
        }
    }

    if (snapshot.activationHistograms !== undefined) {
        patch.activationHistogramBins = snapshot.activationHistograms.bins;
        patch.activationHistogramLayout = {
            binCount: snapshot.activationHistograms.layers[0]?.binCount ?? 0,
            layers: snapshot.activationHistograms.layers,
        };
        patch.activationHistogramProvenance = directArtifact(result, 'activationHistogram', {
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(
                128,
                result.evidence.latestEvaluation!.dataset.trainCount,
            ),
            populationCount: result.evidence.latestEvaluation!.dataset.trainCount,
        });
    } else {
        const hasCurrentHistogram = currentFrame.activationHistogramBins !== null
            || currentFrame.activationHistogramLayout !== null
            || currentFrame.activationHistogramProvenance !== null;
        const canReuseCurrentHistogram = currentFrame.activationHistogramBins !== null
            && currentFrame.activationHistogramLayout !== null
            && currentFrame.activationHistogramProvenance?.model.generationId === result.runId;
        if (replaceAbsentArtifacts || (hasCurrentHistogram && !canReuseCurrentHistogram)) {
            patch.activationHistogramBins = null;
            patch.activationHistogramLayout = null;
            patch.activationHistogramProvenance = null;
        }
    }

    const binaryConfusion = snapshot.testMetrics.confusionMatrix;
    const multiclassConfusion = snapshot.testMetrics.multiclassConfusionMatrix;
    if (binaryConfusion !== undefined && multiclassConfusion !== undefined) {
        throw new Error('direct V2 snapshot cannot contain two confusion matrix kinds');
    }
    if (binaryConfusion !== undefined || multiclassConfusion !== undefined) {
        const pairedConfusion = result.evidence.latestEvaluation?.test.values.confusionMatrix;
        const binaryMatches = binaryConfusion !== undefined
            && pairedConfusion !== undefined
            && !('classCount' in pairedConfusion)
            && binaryConfusion.tp === pairedConfusion.tp
            && binaryConfusion.tn === pairedConfusion.tn
            && binaryConfusion.fp === pairedConfusion.fp
            && binaryConfusion.fn === pairedConfusion.fn;
        const multiclassMatches = multiclassConfusion !== undefined
            && pairedConfusion !== undefined
            && 'classCount' in pairedConfusion
            && multiclassConfusion.classCount === pairedConfusion.classCount
            && multiclassConfusion.classLabels.every(
                (label, index) => label === pairedConfusion.classLabels[index],
            )
            && multiclassConfusion.counts.length === pairedConfusion.counts.length
            && multiclassConfusion.counts.every(
                (count, index) => count === pairedConfusion.counts[index],
            );
        if (!binaryMatches && !multiclassMatches) {
            throw new Error('direct V2 confusion matrix does not match latest paired evaluation');
        }
        patch.confusionMatrix = binaryMatches ? pairedConfusion : null;
        patch.multiclassConfusionMatrix = multiclassMatches ? pairedConfusion : null;
        patch.confusionMatrixProvenance = directArtifact(result, 'confusionMatrix', {
            kind: 'full-split',
            split: 'test',
            sampleCount: result.evidence.latestEvaluation!.dataset.testCount,
            populationCount: result.evidence.latestEvaluation!.dataset.testCount,
        });
        patch.confusionMatrixEvaluationId = result.evidence.latestEvaluation!.evaluationId;
    } else {
        const hasCurrentConfusion = currentFrame.confusionMatrix !== null
            || currentFrame.multiclassConfusionMatrix !== null
            || currentFrame.confusionMatrixProvenance !== null
            || currentFrame.confusionMatrixEvaluationId !== null;
        const canReuseCurrentConfusion = (
            currentFrame.confusionMatrix !== null
            || currentFrame.multiclassConfusionMatrix !== null
        ) && currentFrame.confusionMatrixProvenance?.model.generationId === result.runId;
        if (replaceAbsentArtifacts || (hasCurrentConfusion && !canReuseCurrentConfusion)) {
            patch.confusionMatrix = null;
            patch.multiclassConfusionMatrix = null;
            patch.confusionMatrixProvenance = null;
            patch.confusionMatrixEvaluationId = null;
        }
    }

    const expectedKeys = new Set<keyof WorkerArtifactProvenanceV2>();
    if (hasScalarBoundary || hasMulticlassBoundary) expectedKeys.add('decisionBoundary');
    if (neuronGrids !== null) expectedKeys.add('neuronGrids');
    if (snapshot.layerStats !== undefined) expectedKeys.add('activationStatistics');
    if (snapshot.activationHistograms !== undefined) expectedKeys.add('activationHistogram');
    if (binaryConfusion !== undefined || multiclassConfusion !== undefined) {
        expectedKeys.add('confusionMatrix');
    }
    for (const key of Object.keys(result.artifacts ?? {}) as Array<keyof WorkerArtifactProvenanceV2>) {
        if (!expectedKeys.has(key)) throw new Error(`direct V2 ${key} provenance has no payload`);
    }

    validateFrameBufferPatch(patch, { requireArtifactProvenance: true });
    return patch;
}

const EMPTY_CHECKPOINT_TIMELINE: CheckpointTimeline = {
    checkpoints: [],
    maxCheckpoints: 8,
    evictedCount: 0,
    liveCheckpointId: null,
    restoredCheckpointId: null,
};

export function useTraining(): TrainingHook {
    // All refs first (stable hook order)
    const mountedRef = useRef(true);
    const initializedRef = useRef(false);
    const activePreparedRef = useRef<PreparedExperimentDocumentV2 | null>(null);
    const acknowledgedRunRef = useRef<AcknowledgedRun | null>(null);
    const inFlightRunCommitsRef = useRef(new Map<number, Promise<void>>());
    const prevRunRequestIdRef = useRef<number | null>(null);
    const prevConfigSyncNonceRef = useRef(useTrainingStore.getState().configSyncNonce);
    const requestIdRef = useRef(0);
    const activeRequestIdRef = useRef<number | null>(null);
    const pendingPreparationRequestIdRef = useRef<number | null>(null);
    const rejectedPreparationRequestIdsRef = useRef(new Set<number>());
    const configSyncRejectionRef = useRef<{
        readonly requestId: number;
        readonly reject: (message: string) => void;
    } | null>(null);
    const lifecycleEpochRef = useRef(0);
    const streamSetupPromiseRef = useRef<Promise<void> | null>(null);
    const initializationPromiseRef = useRef<Promise<boolean> | null>(null);
    const mutationPausePromiseRef = useRef<Promise<void> | null>(null);
    const mutationPauseResolveRef = useRef<(() => void) | null>(null);
    const mutationPauseRejectRef = useRef<((error: Error) => void) | null>(null);
    const manualActionPendingRef = useRef(false);
    const stepsPerFrameRef = useRef(5);
    const isPlayingRef = useRef(false);
    const configSyncSeqRef = useRef(0);
    const activeConfigSyncSeqRef = useRef(0);
    const restoreBarrierRef = useRef<Promise<void> | null>(null);
    const e2eStartupFaultActiveRef = useRef(false);

    // Config selectors (from playground store — stable, rarely changes)
    const runRequestId = usePlaygroundStore((s) => s.runRequestId);
    const demand = usePlaygroundStore((s) => s.demand);
    const webgpuGrid = usePlaygroundStore((s) => s.featuresUI.webgpuGrid);

    // Runtime selectors (from training store — volatile)
    const stepsPerFrame = useTrainingStore((s) => s.stepsPerFrame);
    const configSyncNonce = useTrainingStore((s) => s.configSyncNonce);

    const reportWorkerError = useCallback((error: unknown, fallback: string) => {
        if (!mountedRef.current) return;
        const message = getErrorMessage(error, fallback);
        mutationPauseRejectRef.current?.(new Error(message));
        mutationPausePromiseRef.current = null;
        mutationPauseResolveRef.current = null;
        mutationPauseRejectRef.current = null;
        const ts = useTrainingStore.getState();
        isPlayingRef.current = false;
        stopRenderLoop();
        ts.setWorkerError(message);
        ts.setPauseReason('error');
        ts.setStatus('paused');
        initializedRef.current = false;
    }, []);

    const beginConfigSync = useCallback(() => {
        const seq = configSyncSeqRef.current + 1;
        configSyncSeqRef.current = seq;
        activeConfigSyncSeqRef.current = seq;
        useTrainingStore.setState({ runtimeSyncPending: true });
        return seq;
    }, []);

    const isCurrentConfigSync = useCallback((seq: number) => (
        useTrainingStore.getState().runtimeSyncPending && activeConfigSyncSeqRef.current === seq
    ), []);

    const finishConfigSyncIfCurrent = useCallback((seq: number) => {
        if (activeConfigSyncSeqRef.current === seq) {
            useTrainingStore.setState({ runtimeSyncPending: false });
        }
    }, []);

    const nextRequest = useCallback((nextPrepared: PreparedExperimentDocumentV2) => {
        const requestId = requestIdRef.current + 1;
        requestIdRef.current = requestId;
        activeRequestIdRef.current = requestId;
        pendingPreparationRequestIdRef.current = requestId;
        return createWorkerExperimentRequestV2(nextPrepared, requestId);
    }, []);

    const ensureStreamChannel = useCallback((): Promise<void> => {
        if (!streamSetupPromiseRef.current) {
            const promise = setupStreamChannel().catch((error) => {
                if (streamSetupPromiseRef.current === promise) {
                    streamSetupPromiseRef.current = null;
                }
                throw error;
            });
            streamSetupPromiseRef.current = promise;
        }
        return streamSetupPromiseRef.current;
    }, []);

    const pauseForMutation = useCallback((): Promise<void> => {
        if (mutationPausePromiseRef.current) return mutationPausePromiseRef.current;
        if (!isPlayingRef.current) return Promise.resolve();
        isPlayingRef.current = false;
        stopRenderLoop();
        let rejectPause!: (error: Error) => void;
        const promise = new Promise<void>((resolve, reject) => {
            mutationPauseResolveRef.current = resolve;
            mutationPauseRejectRef.current = reject;
            rejectPause = reject;
        });
        mutationPausePromiseRef.current = promise;
        try {
            postStreamCommand({ type: 'stopTraining', protocolVersion: WORKER_PROTOCOL_VERSION });
        } catch (error) {
            mutationPausePromiseRef.current = null;
            mutationPauseResolveRef.current = null;
            mutationPauseRejectRef.current = null;
            rejectPause(error instanceof Error ? error : new Error(String(error)));
        }
        return promise;
    }, []);

    const trackWorkerRunCommit = useCallback((
        resultPromise: Promise<InitializeExperimentResult>,
        requestId: number,
        lifecycleEpoch: number,
        owner: PreparedExperimentDocumentV2,
        source: TrainedRecipeSource,
    ): Promise<AcknowledgedRun | null> => {
        const committed = Promise.resolve(resultPromise).then((result) => {
            if (!mountedRef.current || lifecycleEpochRef.current !== lifecycleEpoch
                || rejectedPreparationRequestIdsRef.current.has(requestId)) return null;
            const { evidenceGenerationId } = useTrainingStore.getState();
            if (result.runId <= Math.max(acknowledgedRunRef.current?.runId ?? 0, evidenceGenerationId ?? 0)) {
                return null;
            }
            // Even a superseded request may have committed on the worker.
            // Validate it now and retain it as the fallback for a newer rejection.
            const evidence = parseWorkerEvidenceMessageV2(result.evidence);
            const dataset = evidence.latestEvaluation?.dataset;
            if (!dataset) throw new Error('fresh V2 result requires dataset identity');
            const run = {
                runId: result.runId,
                owner,
                source,
                dataset,
                evidence,
                framePatch: buildStrictV2FramePatch(result, true, owner),
                timeline: parseCheckpointTimelineV2(result.checkpointTimeline),
            };
            acknowledgedRunRef.current = run;
            usePlaygroundStore.getState().acknowledgeRuntimeDocument(owner);
            return run;
        });
        const settled = committed.then(() => {}, () => {});
        const inFlight = inFlightRunCommitsRef.current;
        inFlight.set(requestId, settled);
        void settled.then(() => inFlight.delete(requestId));
        return committed;
    }, []);

    const publishCommittedV2Run = useCallback((run: AcknowledgedRun, publishAccess = true): void => {
        const ts = useTrainingStore.getState();
        if (run.runId <= (ts.evidenceGenerationId ?? 0)) {
            throw new Error('fresh V2 result must advance the active generation monotonically');
        }
        const evidenceReplacement = ts.prepareEvidenceReplacement(run.evidence);
        newRunTo(run.runId);
        ts.commitEvidenceReplacement(evidenceReplacement);
        updateFrameBuffer(run.framePatch, { requireArtifactProvenance: true });
        ts.setFrameVersions(getFrameVersions());
        activePreparedRef.current = run.owner;
        ts.setCheckpointTimeline(run.timeline);
        ts.clearWorkerError();
        ts.clearPauseReason();
        ts.markTrainedRecipe(
            run.owner.document.recipe,
            run.source,
            run.owner.identities.recipeFingerprint,
        );
        ts.setStatus('idle');
        initializedRef.current = false;
        if (publishAccess) usePlaygroundStore.getState().acknowledgeRuntimeDocument(run.owner, true);
    }, []);

    const hydrateCommittedV2Run = useCallback(async (
        api: TrainingWorkerRemote,
        runId: number,
        dataset: DatasetRevision,
        isCurrent: () => boolean,
    ): Promise<boolean> => {
        const points = await api.getDatasetPointsV2(runId, dataset.datasetKey);
        if (!isCurrent()) return false;
        validateDatasetPoints(points, runId, dataset);

        await api.updateDemand(usePlaygroundStore.getState().demand);
        if (!isCurrent()) return false;
        try {
            await api.setWebGpuEnabled(usePlaygroundStore.getState().featuresUI.webgpuGrid);
        } catch {
            // The worker can fall back to CPU grids if WebGPU is unavailable.
        }
        if (!isCurrent()) return false;

        initializedRef.current = useTrainingStore.getState().hydrateDataset(
            runId, points.trainPoints, points.testPoints,
        );
        return initializedRef.current;
    }, []);

    const initializePrepared = useCallback(async (
        requestedPrepared: PreparedExperimentDocumentV2,
        requestedRunRequestId: number,
    ): Promise<boolean> => {
        const lifecycleEpoch = lifecycleEpochRef.current;
        prevRunRequestIdRef.current = requestedRunRequestId;
        const request = nextRequest(requestedPrepared);
        const isCurrent = () => mountedRef.current
            && lifecycleEpochRef.current === lifecycleEpoch
            && activeRequestIdRef.current === request.requestId
            && isCurrentRunRequest(requestedRunRequestId, requestedPrepared);
        const ts = useTrainingStore.getState();
        if (ts.datasetStatus === 'error') ts.setDatasetLoading();
        try {
            const api = await getWorkerApi();
            await ensureStreamChannel();
            if (!isCurrent()) return false;
            const run = await trackWorkerRunCommit(
                api.initializeExperimentV2(request), request.requestId, lifecycleEpoch, requestedPrepared, 'initialize',
            );
            if (rejectedPreparationRequestIdsRef.current.delete(request.requestId) || !isCurrent()) return false;
            pendingPreparationRequestIdRef.current = null;
            if (!run) throw new Error('fresh V2 result must advance the active generation monotonically');
            publishCommittedV2Run(run);
            return await hydrateCommittedV2Run(api, run.runId, run.dataset, isCurrent);
        } catch (error) {
            rejectedPreparationRequestIdsRef.current.delete(request.requestId);
            if (!isCurrent()) return false;
            pendingPreparationRequestIdRef.current = null;
            throw error;
        }
    }, [ensureStreamChannel, hydrateCommittedV2Run, nextRequest, publishCommittedV2Run, trackWorkerRunCommit]);

    const initializeWorker = useCallback((
        requestedPrepared: PreparedExperimentDocumentV2 = activePreparedRef.current ?? requirePreparedExperiment(),
    ): Promise<boolean> => {
        const requestedRunRequestId = usePlaygroundStore.getState().runRequestId;
        const inFlight = initializationPromiseRef.current;
        if (inFlight && prevRunRequestIdRef.current === requestedRunRequestId) return inFlight;
        const promise = initializePrepared(requestedPrepared, requestedRunRequestId);
        initializationPromiseRef.current = promise;
        const clear = () => {
            if (initializationPromiseRef.current === promise) {
                initializationPromiseRef.current = null;
            }
        };
        void promise.then(clear, clear);
        return promise;
    }, [initializePrepared]);

    // Keep ref in sync so streaming commands use current speed.
    useEffect(() => {
        stepsPerFrameRef.current = stepsPerFrame;
        // If currently playing, update the worker's speed
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'updateSpeed', protocolVersion: WORKER_PROTOCOL_VERSION, stepsPerFrame });
        }
    }, [stepsPerFrame]);

    // ── Snapshot handler: applies streamed snapshots to training store ──
    useEffect(() => {
        const unsubscribe = onSnapshot((msg: WorkerToMainMessage) => {
            if (!mountedRef.current) return;
            const ts = useTrainingStore.getState();

            if (msg.type === 'evidence') {
                try {
                    const activePrepared = activePreparedRef.current;
                    if (activePrepared === null) {
                        throw new Error('scientific evidence arrived before an active prepared run');
                    }
                    ts.applyEvidence(validateEvidenceAgainstPrepared(msg, activePrepared));
                } catch (error) {
                    reportWorkerError(error, 'Received invalid scientific evidence from the worker.');
                }
            } else if (msg.type === 'worker-error') {
                const error = msg as WorkerProtocolErrorMessageV2;
                if (error.generationId === null) {
                    if (error.requestId === null
                        || error.requestId !== pendingPreparationRequestIdRef.current) return;
                    pendingPreparationRequestIdRef.current = null;
                    rejectedPreparationRequestIdsRef.current.add(error.requestId);
                    const configSyncRejection = configSyncRejectionRef.current;
                    if (configSyncRejection?.requestId === error.requestId) {
                        configSyncRejection.reject(error.message);
                        return;
                    }
                    activeRequestIdRef.current = null;
                    reportWorkerError(error.message, 'Failed to prepare the experiment.');
                    return;
                }
                if (error.generationId !== ts.evidenceGenerationId) return;
                reportWorkerError(error.message, 'The training worker failed.');
            } else if (msg.type === 'snapshot') {
                try {
                    const checkpointTimeline = parseCheckpointTimelineV2(msg.checkpointTimeline);
                    const frameVersions = getFrameVersions();
                    ts.applyStreamedFrame({
                        frameVersions,
                        checkpointTimeline,
                    });
                } catch (error) {
                    reportWorkerError(error, 'Received invalid checkpoint metadata from the worker.');
                }
            } else if (msg.type === 'status') {
                if (msg.status === 'paused') {
                    if (msg.checkpointTimeline !== undefined) {
                        try {
                            ts.applyStreamedFrame({
                                frameVersions: getFrameVersions(),
                                checkpointTimeline: parseCheckpointTimelineV2(msg.checkpointTimeline),
                            });
                        } catch (error) {
                            reportWorkerError(
                                error,
                                'Received invalid checkpoint metadata from the worker.',
                            );
                            return;
                        }
                    }
                    const resolveMutationPause = mutationPauseResolveRef.current;
                    if (resolveMutationPause) {
                        mutationPausePromiseRef.current = null;
                        mutationPauseResolveRef.current = null;
                        mutationPauseRejectRef.current = null;
                        resolveMutationPause();
                    }
                    if (msg.pauseReason) {
                        isPlayingRef.current = false;
                        stopRenderLoop();
                        ts.setPauseReason(msg.pauseReason);
                    }
                    ts.setStatus('paused');
                } else if (msg.status === 'idle') {
                    isPlayingRef.current = false;
                    stopRenderLoop();
                    ts.clearPauseReason();
                    ts.setStatus(msg.status);
                } else if (msg.status === 'running') {
                    ts.clearPauseReason();
                    ts.setStatus('running');
                }
            } else if (msg.type === 'error') {
                reportWorkerError(msg.message, 'The training worker failed.');
            }
        });

        return unsubscribe;
    }, [reportWorkerError]);

    // Initialize worker on mount
    useEffect(() => {
        lifecycleEpochRef.current++;
        mountedRef.current = true;
        if (e2eStartupFaultActiveRef.current) return;
        const prepared = currentPreparedExperiment();
        if (!prepared) {
            reportWorkerError(
                new Error('Training is unavailable because the shared experiment URL is incompatible with version 2.'),
                'Failed to initialize training worker.',
            );
            return;
        }
        // Build-time guard: production bundles drop the fault-injection helpers.
        if (import.meta.env.VITE_E2E_FAULTS === '1') {
            const fault = consumeE2EWorkerFault(
                new URL(window.location.href),
                window.sessionStorage,
                true,
            );
            if (fault === 'startup-once') {
                e2eStartupFaultActiveRef.current = true;
                queueMicrotask(() => emitE2EWorkerError(E2E_WORKER_FAULT_MESSAGE));
                return;
            }
        }
        initializeWorker(prepared).catch((error) => {
            reportWorkerError(error, 'Failed to initialize training worker.');
        });
        // Mount initialization is intentionally one-shot. Later prepared
        // documents are handled by the ordered config transaction below.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // An explicit run request starts a fresh generation even for an identical recipe.
    useEffect(() => {
        if (e2eStartupFaultActiveRef.current) return;
        const retryRequested = configSyncNonce !== prevConfigSyncNonceRef.current;
        const visiblePrepared = currentPreparedExperiment();
        if (!visiblePrepared) return;
        if (!retryRequested && prevRunRequestIdRef.current === runRequestId) return;
        prevRunRequestIdRef.current = runRequestId;
        prevConfigSyncNonceRef.current = configSyncNonce;
        const ts = useTrainingStore.getState();
        const rejected = ts.rejectedConfigRequest;
        const retry = retryRequested && rejected?.runRequestId === runRequestId ? rejected : null;
        const prepared = retry ? {
            ...retry.prepared,
            document: { ...retry.prepared.document, view: visiblePrepared.document.view },
        } : visiblePrepared;
        const requestSource = retry?.runRequestSource ?? usePlaygroundStore.getState().runRequestSource;
        const candidate: RejectedConfigRequest = { runRequestId, runRequestSource: requestSource, prepared };
        const source = requestSource === 'document' ? ts.pendingConfigSource ?? 'setup' : requestSource;
        if (ts.pendingConfigSource !== source) ts.beginConfigChange(source);
        const seq = beginConfigSync();
        const request = nextRequest(prepared);
        const lifecycleEpoch = lifecycleEpochRef.current;
        const isCurrent = () => mountedRef.current
            && lifecycleEpochRef.current === lifecycleEpoch
            && isCurrentConfigSync(seq)
            && activeRequestIdRef.current === request.requestId
            && isCurrentRunRequest(runRequestId, visiblePrepared);

        let rejecting = false;
        const reject = (message: string) => {
            if (rejecting || !isCurrent()) return;
            rejecting = true;
            const recover = () => {
                if (!isCurrent()) return;
                const acknowledged = acknowledgedRunRef.current;
                if (acknowledged && acknowledged.runId > (useTrainingStore.getState().evidenceGenerationId ?? 0)) {
                    publishCommittedV2Run(acknowledged, false);
                }
                ts.failConfigChange(message, candidate);
                if (acknowledged) {
                    const playground = usePlaygroundStore.getState();
                    playground.restoreAcknowledgedDocument(acknowledged.owner);
                    playground.syncToUrl();
                }
                if (!acknowledged || initializedRef.current) {
                    finishConfigSyncIfCurrent(seq);
                    return;
                }
                const recoveryIsCurrent = () => mountedRef.current
                    && lifecycleEpochRef.current === lifecycleEpoch
                    && isCurrentConfigSync(seq)
                    && activeRequestIdRef.current === request.requestId
                    && usePlaygroundStore.getState().runRequestId === runRequestId;
                void getWorkerApi().then((api) => hydrateCommittedV2Run(
                    api, acknowledged.runId, acknowledged.dataset, recoveryIsCurrent,
                )).catch((error) => {
                    if (recoveryIsCurrent()) reportWorkerError(error, 'Failed to hydrate the previous experiment runtime.');
                }).finally(() => finishConfigSyncIfCurrent(seq));
            };
            // RPC responses for earlier worker commits and the current stream
            // error arrive on separate channels. Reconcile with the newest
            // validated commit before handing control back to the user.
            const earlier = [...inFlightRunCommitsRef.current]
                .filter(([requestId]) => requestId < request.requestId
                    && !rejectedPreparationRequestIdsRef.current.has(requestId))
                .map(([, settled]) => settled);
            if (earlier.length > 0) void Promise.all(earlier).then(recover);
            else recover();
        };
        configSyncRejectionRef.current = { requestId: request.requestId, reject };

        const sync = async () => {
            const restoreBarrier = restoreBarrierRef.current;
            if (restoreBarrier) {
                await restoreBarrier;
                if (!isCurrent()) return;
            }
            // Serialize MessagePort pause with the Comlink mutation. The
            // paused status is the acknowledgement that the old loop and its
            // forced evaluation have completed.
            try {
                await pauseForMutation();
            } catch (error) {
                if (isCurrent()) pendingPreparationRequestIdRef.current = null;
                reject(getErrorMessage(error, 'Failed to pause the current experiment.'));
                return;
            }
            if (!isCurrent()) return;

            let api: TrainingWorkerRemote;
            let run: AcknowledgedRun;
            try {
                api = await getWorkerApi();
                await ensureStreamChannel();
                if (!isCurrent()) return;
                const committed = await trackWorkerRunCommit(
                    api.initializeExperimentV2(request), request.requestId, lifecycleEpoch, prepared, 'config-sync',
                );
                if (rejectedPreparationRequestIdsRef.current.delete(request.requestId) || !isCurrent()) return;
                pendingPreparationRequestIdRef.current = null;
                if (!committed) throw new Error('fresh V2 result must advance the active generation monotonically');
                run = committed;
                publishCommittedV2Run(run);
            } catch (error) {
                rejectedPreparationRequestIdsRef.current.delete(request.requestId);
                if (!isCurrent()) return;
                pendingPreparationRequestIdRef.current = null;
                reject(getErrorMessage(error, 'Failed to update configuration.'));
                return;
            }

            usePlaygroundStore.getState().syncToUrl();
            const publicationIsCurrent = () => mountedRef.current
                && lifecycleEpochRef.current === lifecycleEpoch
                && isCurrentConfigSync(seq)
                && activeRequestIdRef.current === request.requestId
                && isCurrentRunRequest(runRequestId, prepared);
            try {
                await hydrateCommittedV2Run(api, run.runId, run.dataset, publicationIsCurrent);
            } catch (error) {
                if (publicationIsCurrent()) {
                    reportWorkerError(error, 'Failed to hydrate the new experiment runtime.');
                }
            }
            if (!publicationIsCurrent()) return;
            ts.finishConfigChange();
            finishConfigSyncIfCurrent(seq);
        };
        void sync();
    }, [
        runRequestId,
        configSyncNonce,
        beginConfigSync,
        finishConfigSyncIfCurrent,
        ensureStreamChannel,
        hydrateCommittedV2Run,
        isCurrentConfigSync,
        nextRequest,
        pauseForMutation,
        publishCommittedV2Run,
        reportWorkerError,
        trackWorkerRunCommit,
    ]);

    // Sync demand changes to worker
    useEffect(() => {
        if (!initializedRef.current) return;
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'updateDemand', protocolVersion: WORKER_PROTOCOL_VERSION, demand });
            return;
        }
        void getWorkerApi()
            .then((api) => api.updateDemand(demand))
            .catch((error: unknown) => {
                reportWorkerError(error, 'Failed to update visualization demand.');
            });
    }, [demand, reportWorkerError]);

    // AS-4: live-toggle the WebGPU grid path when the user flips the
    // featuresUI flag. Disabling immediately disposes the GPU predictor
    // (frees device memory); enabling lets the next snapshot lazily
    // re-allocate.
    useEffect(() => {
        if (!initializedRef.current) return;
        void getWorkerApi()
            .then((api) => api.setWebGpuEnabled(webgpuGrid))
            .catch(() => {
                // Ignore — capability detection inside the worker handles
                // any per-device fallback. A toggle that doesn't reach the
                // worker just means the next snapshot still uses whatever
                // path the worker last knew about.
            });
    }, [webgpuGrid]);

    const play = useCallback(() => {
        const training = useTrainingStore.getState();
        if (useTrainingStore.getState().runtimeSyncPending
            || mutationPausePromiseRef.current !== null
            || manualActionPendingRef.current
            || training.pendingConfigSource !== null
            || selectCheckpointBranchComplete(training)) {
            return;
        }

        const startTraining = () => {
            isPlayingRef.current = true;
            const ts = useTrainingStore.getState();
            ts.clearPauseReason();
            ts.setStatus('running');
            startRenderLoop();
            postStreamCommand({
                type: 'startTraining',
                protocolVersion: WORKER_PROTOCOL_VERSION,
                stepsPerFrame: stepsPerFrameRef.current,
            });
        };

        if (!initializedRef.current) {
            void initializeWorker().then((ready) => {
                if (ready && mountedRef.current && !useTrainingStore.getState().runtimeSyncPending
                    && useTrainingStore.getState().pendingConfigSource === null) {
                    startTraining();
                }
            }).catch((error) => {
                reportWorkerError(error, 'Failed to initialize training worker.');
            });
            return;
        }
        startTraining();
    }, [initializeWorker, reportWorkerError]);

    const pause = useCallback(() => {
        if (!isPlayingRef.current && useTrainingStore.getState().status !== 'running') {
            return;
        }
        const ts = useTrainingStore.getState();
        ts.setPauseReason('manual');
        ts.setStatus('paused');
        void pauseForMutation().catch((error) => {
            reportWorkerError(error, 'Failed to pause training.');
        });
    }, [pauseForMutation, reportWorkerError]);

    const step = useCallback(async () => {
        const training = useTrainingStore.getState();
        if (training.runtimeSyncPending || training.pendingConfigSource !== null
            || selectCheckpointBranchComplete(training)) {
            return;
        }
        if (manualActionPendingRef.current) return;
        manualActionPendingRef.current = true;
        const wasPlaying = isPlayingRef.current;
        try {
            await pauseForMutation();
            if (useTrainingStore.getState().runtimeSyncPending
                || useTrainingStore.getState().pendingConfigSource !== null) return;
            if (wasPlaying) {
                useTrainingStore.getState().setPauseReason('manual');
                useTrainingStore.getState().setStatus('paused');
            }
            if (!initializedRef.current) {
                const ready = await initializeWorker();
                if (!ready) return;
            }
            const api = await getWorkerApi();
            const result = await api.stepExperimentV2(1, 'current');
            if (!mountedRef.current) return;
            const ts = useTrainingStore.getState();
            if (ts.runtimeSyncPending
                || ts.pendingConfigSource !== null
                || result.runId !== ts.evidenceGenerationId) return;
            const resultRevision = selectScientificEvidence({
                latestLiveSignal: result.evidence.liveSignal ?? null,
                latestEvaluation: result.evidence.latestEvaluation ?? null,
            }).currentModel?.revision;
            const currentRevision = selectScientificEvidence({
                latestLiveSignal: ts.latestLiveSignal,
                latestEvaluation: ts.latestEvaluation,
            }).currentModel?.revision;
            if (resultRevision !== undefined
                && currentRevision !== undefined
                && resultRevision < currentRevision) return;
            const expectedPrepared = activePreparedRef.current;
            if (expectedPrepared === null) {
                throw new Error('direct V2 step requires an active prepared experiment');
            }
            const framePatch = buildStrictV2FramePatch(
                result,
                false,
                expectedPrepared,
            );
            const directEvidence = evidenceForDirectStep(
                result.evidence,
                ts.latestEvaluation,
            );
            const preparedEvidence = directEvidence === null
                ? null
                : ts.prepareEvidenceAppend(
                    validateEvidenceAgainstPrepared(directEvidence, expectedPrepared),
                );
            updateFrameBuffer(framePatch, { requireArtifactProvenance: true });
            const frameVersions = getFrameVersions();
            ts.setFrameVersions(frameVersions);
            if (preparedEvidence !== null) ts.commitEvidenceAppend(preparedEvidence);
            ts.setCheckpointTimeline(parseCheckpointTimelineV2(result.checkpointTimeline));
        } catch (error) {
            reportWorkerError(error, 'Failed to run a training step.');
        } finally {
            manualActionPendingRef.current = false;
        }
    }, [initializeWorker, pauseForMutation, reportWorkerError]);

    const reset = useCallback(async () => {
        if (useTrainingStore.getState().runtimeSyncPending
            || manualActionPendingRef.current
            || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        const seq = beginConfigSync();
        const resetRunRequestId = usePlaygroundStore.getState().runRequestId;
        const lifecycleEpoch = lifecycleEpochRef.current;
        const resetPrepared = activePreparedRef.current ?? currentPreparedExperiment();
        const isCurrent = () => mountedRef.current && isCurrentConfigSync(seq)
            && lifecycleEpochRef.current === lifecycleEpoch
            && (resetPrepared === null || isCurrentRunRequest(resetRunRequestId, resetPrepared));
        useTrainingStore.getState().clearPauseReason();
        try {
            await pauseForMutation();
            if (!isCurrent()) return;
            if (!initializedRef.current) {
                const ready = await initializeWorker();
                if (ready && isCurrent()) useTrainingStore.getState().setStatus('idle');
                return;
            }
            const api = await getWorkerApi();
            if (!isCurrent() || !resetPrepared) return;
            const requestId = ++requestIdRef.current;
            const run = await trackWorkerRunCommit(
                api.resetExperimentV2(), requestId, lifecycleEpoch, resetPrepared, 'reset',
            );
            if (!isCurrent()) return;
            if (!run) throw new Error('fresh V2 result must advance the active generation monotonically');
            publishCommittedV2Run(run);
            await hydrateCommittedV2Run(api, run.runId, run.dataset, isCurrent);
        } catch (error) {
            if (isCurrent()) reportWorkerError(error, 'Failed to reset training.');
        } finally {
            finishConfigSyncIfCurrent(seq);
        }
    }, [beginConfigSync, finishConfigSyncIfCurrent, hydrateCommittedV2Run, initializeWorker, isCurrentConfigSync, pauseForMutation, publishCommittedV2Run, reportWorkerError, trackWorkerRunCommit]);

    const applyCheckpoint = useCallback(async (
        id: number,
        action: 'restore' | 'branch',
        trainingSteps?: number,
    ) => {
        if (!Number.isSafeInteger(id) || id < 1) return;
        if (trainingSteps !== undefined && !isCheckpointBranchStepCount(trainingSteps)) {
            throw new RangeError('Choose a whole number of training steps within the branch limit.');
        }
        if (useTrainingStore.getState().runtimeSyncPending
            || manualActionPendingRef.current
            || restoreBarrierRef.current !== null
            || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        if (!useTrainingStore.getState().checkpointTimeline.checkpoints.some(
            (checkpoint) => checkpoint.id === id,
        )) return;
        manualActionPendingRef.current = true;
        let releaseRestore!: () => void;
        const restoreBarrier = new Promise<void>((resolve) => {
            releaseRestore = resolve;
        });
        restoreBarrierRef.current = restoreBarrier;
        const lifecycleEpoch = lifecycleEpochRef.current;
        try {
            await pauseForMutation();
            if (!mountedRef.current || useTrainingStore.getState().runtimeSyncPending) return;
            if (!initializedRef.current) {
                await initializeWorker();
                if (!initializedRef.current) return;
            }
            const restoredPrepared = activePreparedRef.current;
            const before = useTrainingStore.getState();
            const currentGeneration = before.evidenceGenerationId;
            const currentModel = selectScientificEvidence({
                latestLiveSignal: before.latestLiveSignal,
                latestEvaluation: before.latestEvaluation,
            }).currentModel;
            if (currentGeneration === null || !currentModel || !restoredPrepared) {
                throw new Error('applying a checkpoint requires current scientific evidence');
            }
            const requestId = requestIdRef.current + 1;
            requestIdRef.current = requestId;
            const api = await getWorkerApi();
            const request = {
                protocolVersion: WORKER_PROTOCOL_VERSION,
                requestId,
                checkpointId: id,
            };
            const result = action === 'branch'
                ? await api.branchCheckpointV2({
                    ...request, type: 'branch-checkpoint',
                    ...(trainingSteps === undefined ? {} : { trainingSteps }),
                })
                : await api.restoreCheckpointV2({ ...request, type: 'restore-checkpoint' });
            if (!mountedRef.current
                || lifecycleEpochRef.current !== lifecycleEpoch
                || activePreparedRef.current !== restoredPrepared) {
                return;
            }
            const ts = useTrainingStore.getState();
            const evaluation = result.evidence.latestEvaluation;
            if (result.runId !== currentGeneration
                || evaluation?.model.generationId !== currentGeneration
                || evaluation.model.revision !== currentModel.revision + 1) {
                throw new Error('applying a checkpoint must advance the current revision exactly once');
            }
            const timeline = parseCheckpointTimelineV2(result.checkpointTimeline);
            const appliedCheckpointId = action === 'branch'
                ? timeline.branch?.sourceCheckpointId
                : timeline.restoredCheckpointId;
            if (appliedCheckpointId !== id) {
                throw new Error(`checkpoint timeline does not identify the ${action} source`);
            }
            if (action === 'branch') {
                const source = timeline.checkpoints.find((point) => point.id === id);
                const previous = timeline.checkpoints.find((point) => point.id === timeline.branch?.previousCheckpointId);
                const expectedTarget = trainingSteps === undefined ? undefined : evaluation.model.step + trainingSteps;
                if (source?.step !== evaluation.model.step || source.epoch !== evaluation.model.epoch
                    || previous?.step !== currentModel.step || previous.epoch !== currentModel.epoch
                    || timeline.branch?.targetStep !== expectedTarget) {
                    throw new Error('checkpoint branch timeline does not match the source and previous models');
                }
            }
            const evidence = validateEvidenceAgainstPrepared(result.evidence, restoredPrepared);
            const evidenceReplacement = ts.prepareEvidenceReplacement(evidence);
            const framePatch = buildStrictV2FramePatch(
                result,
                true,
                restoredPrepared,
                'restore',
            );

            discardPendingSnapshot(result.runId, evaluation.model.revision);
            ts.commitEvidenceReplacement(evidenceReplacement);
            updateFrameBuffer(framePatch, { requireArtifactProvenance: true });
            const frameVersions = getFrameVersions();
            ts.setFrameVersions(frameVersions);
            ts.setCheckpointTimeline(timeline);
            ts.markTrainedRecipe(
                restoredPrepared.document.recipe,
                'restore',
                restoredPrepared.identities.recipeFingerprint,
            );
            ts.setPauseReason('manual');
            ts.setStatus('paused');
            ts.clearWorkerError();
        } catch (error) {
            reportWorkerError(error, action === 'branch'
                ? 'Failed to branch from checkpoint.'
                : 'Failed to restore checkpoint.');
            if (action === 'branch') throw error;
        } finally {
            if (restoreBarrierRef.current === restoreBarrier) {
                restoreBarrierRef.current = null;
            }
            manualActionPendingRef.current = false;
            releaseRestore();
        }
    }, [initializeWorker, pauseForMutation, reportWorkerError]);

    const restoreCheckpoint = useCallback((id: number) => applyCheckpoint(id, 'restore'), [applyCheckpoint]);
    const branchCheckpoint = useCallback((id: number, trainingSteps?: number) => (
        applyCheckpoint(id, 'branch', trainingSteps)
    ), [applyCheckpoint]);

    // Cleanup on unmount
    useEffect(() => {
        const lifecycleEpoch = lifecycleEpochRef;
        const rejectedPreparationRequestIds = rejectedPreparationRequestIdsRef;
        const inFlightRunCommits = inFlightRunCommitsRef;
        return () => {
            mountedRef.current = false;
            lifecycleEpoch.current++;
            activePreparedRef.current = null;
            acknowledgedRunRef.current = null;
            inFlightRunCommits.current.clear();
            activeRequestIdRef.current = null;
            pendingPreparationRequestIdRef.current = null;
            rejectedPreparationRequestIds.current.clear();
            configSyncRejectionRef.current = null;
            streamSetupPromiseRef.current = null;
            initializationPromiseRef.current = null;
            mutationPauseRejectRef.current?.(new Error('Training hook unmounted'));
            mutationPausePromiseRef.current = null;
            mutationPauseResolveRef.current = null;
            mutationPauseRejectRef.current = null;
            activeConfigSyncSeqRef.current = configSyncSeqRef.current + 1;
            configSyncSeqRef.current = activeConfigSyncSeqRef.current;
            manualActionPendingRef.current = false;
            isPlayingRef.current = false;
            initializedRef.current = false;
            terminateWorker();
            const ts = useTrainingStore.getState();
            ts.resetEvidence();
            const frameVersions = getFrameVersions();
            useTrainingStore.setState({
                status: 'idle',
                frameVersion: frameVersions.frameVersion,
                outputGridVersion: frameVersions.outputGridVersion,
                neuronGridsVersion: frameVersions.neuronGridsVersion,
                paramsVersion: frameVersions.paramsVersion,
                layerStatsVersion: frameVersions.layerStatsVersion,
                confusionMatrixVersion: frameVersions.confusionMatrixVersion,
                activationHistogramsVersion: frameVersions.activationHistogramsVersion,
                multiclassBoundaryVersion: frameVersions.multiclassBoundaryVersion,
                dataConfigLoading: false,
                networkConfigLoading: false,
                featuresConfigLoading: false,
                trainingConfigLoading: false,
                presetConfigLoading: false,
                pendingConfigSource: null,
                runtimeSyncPending: false,
                configError: null,
                configErrorSource: null,
                rejectedConfigRequest: null,
                workerError: null,
                pauseReason: null,
                checkpointTimeline: EMPTY_CHECKPOINT_TIMELINE,
                trainedRecipe: null,
                trainedRecipeFingerprint: null,
                trainedRecipeRecordedAt: null,
                trainedRecipeSource: null,
            });
        };
    }, []);

    return { play, pause, step, reset, restoreCheckpoint, branchCheckpoint };
}
