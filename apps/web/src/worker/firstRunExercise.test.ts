import { afterEach, expect, it, vi } from 'vitest';
import {
    DEFAULT_DEMAND,
    DEFAULT_EXPERIMENT_DOCUMENT,
    FIRST_RUN_EXERCISE_RECIPE,
    WORKER_PROTOCOL_VERSION,
    prepareExperimentDocument,
} from '@nn-playground/shared';

vi.mock('comlink', () => ({ expose: vi.fn() }));
afterEach(() => vi.unstubAllGlobals());

it('makes the seeded first exercise prediction field visibly change in three manual steps', async () => {
    vi.stubGlobal('postMessage', vi.fn());
    const { workerApi } = await import('./training.worker.ts');
    const prepared = await prepareExperimentDocument({
        ...DEFAULT_EXPERIMENT_DOCUMENT,
        recipe: FIRST_RUN_EXERCISE_RECIPE,
    });
    if (!prepared.ok) throw new Error(JSON.stringify(prepared.issues));
    workerApi.updateDemand(DEFAULT_DEMAND);
    const initial = await workerApi.initializeExperimentV2({
        type: 'initialize-experiment',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId: 1,
        document: prepared.value.document,
        claimedIdentities: prepared.value.identities,
    });
    const pointCount = initial.snapshot.gridSize ** 2;
    expect(initial.snapshot.outputGrid).toHaveLength(pointCount);
    expect(Array.from(initial.snapshot.outputGrid).every((probability) => probability === 0.5)).toBe(true);

    await workerApi.stepExperimentV2(1, 'current');
    await workerApi.stepExperimentV2(1, 'current');
    const stepped = await workerApi.stepExperimentV2(1, 'current');
    const grid = Array.from(stepped.snapshot.outputGrid);
    expect(stepped.artifacts?.decisionBoundary?.model.step).toBe(3);
    expect(grid).toHaveLength(pointCount);
    expect(grid.filter((probability) => probability < 0.3).length).toBeGreaterThan(pointCount * 0.1);
    expect(grid.filter((probability) => probability > 0.7).length).toBeGreaterThan(pointCount * 0.1);
    expect(stepped.evidence.latestEvaluation!.test.values.dataLoss)
        .toBeLessThan(initial.evidence.latestEvaluation!.test.values.dataLoss - 0.1);
});
