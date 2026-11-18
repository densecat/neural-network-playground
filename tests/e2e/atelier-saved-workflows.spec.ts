import { readFile } from 'node:fs/promises';
import { expect, test, type Locator, type Page } from '@playwright/test';
import type { ExperimentRunRecordV2 } from '../../packages/shared/src/experimentMemory';
import { applyDataset, expectEvidence, identity, learning, ready, transport } from './atelier-helpers';

const setupSection = (page: Page, name: string) => page.getByRole('navigation', { name: 'Setup sections' }).getByRole('button', { name, exact: true });
const runName = (page: Page) => page.getByRole('textbox', { name: 'Run name', exact: true });
const savedRun = (page: Page, title: string) => page.getByRole('article', { name: title, exact: true });

async function openActions(page: Page, title: string) {
    const row = page.getByRole('row').filter({ has: savedRun(page, title) });
    const details = row.locator('details');
    if (await details.getAttribute('open') === null) await details.locator('summary').click();
    return details;
}

async function save(page: Page, title: string) {
    await runName(page).fill(title);
    await page.getByRole('button', { name: 'Save current run', exact: true }).click();
    await expect(savedRun(page, title)).toBeVisible();
}

async function stepWithEvidence(page: Page, step: number) {
    await transport(page).getByRole('button', { name: 'Run one training step', exact: true }).click();
    await expectEvidence(page, step);
}

async function downloadFile(page: Page, button: Locator) {
    const pending = page.waitForEvent('download');
    await button.click();
    const download = await pending;
    return { name: download.suggestedFilename(), text: await readFile((await download.path())!, 'utf8') };
}

test.beforeEach(async ({ page }) => {
    await page.goto('./');
    await ready(page);
});

test('saved evaluation CSV matches the complete saved history after further training', async ({ page }) => {
    await learning(page);
    await stepWithEvidence(page, 1);
    await stepWithEvidence(page, 2);
    const original = await identity(page);
    const saveRun = page.getByRole('button', { name: 'Save run…', exact: true });
    await saveRun.press('Enter');
    await expect(runName(page)).toBeFocused();
    await save(page, 'CSV reference');
    expect(await identity(page)).toEqual(original);

    await learning(page);
    await stepWithEvidence(page, 3);
    const later = await identity(page);
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    const actions = await openActions(page, 'CSV reference');
    const json = await downloadFile(page, actions.getByRole('button', { name: 'Download evidence for CSV reference', exact: true }));
    const csv = await downloadFile(page, actions.getByRole('button', { name: 'Download evaluation CSV', exact: true }));
    const record = JSON.parse(json.text) as ExperimentRunRecordV2;
    expect(record.snapshot.model.step).toBe(2);
    expect(csv.name).toBe(`${record.id}-evaluation.csv`);
    const [header, ...lines] = csv.text.trimEnd().split(/\r?\n/);
    expect(header.split(',')).toEqual([
        'series', 'step', 'revision', 'train_sample_count', 'test_sample_count',
        'train_data_loss', 'test_data_loss', 'train_total_objective', 'batch_sample_count', 'batch_ema_data_loss',
    ]);
    const rows = lines.map((line) => line.split(','));
    const full = rows.filter(([series]) => series === 'full-evaluation');
    const batch = rows.filter(([series]) => series === 'mini-batch-ema');
    expect(full.length).toBeGreaterThan(1);
    expect(batch.length).toBeGreaterThan(0);
    expect(rows).toHaveLength(full.length + batch.length);
    expect(full).toEqual(record.snapshot.evaluationHistory.map(({ model, train, test, objective }) => [
        'full-evaluation', model.step, model.revision, train.basis.sampleCount, test.basis.sampleCount,
        train.values.dataLoss, test.values.dataLoss, objective.trainTotalObjective, '', '',
    ].map(String)));
    expect(batch).toEqual(record.snapshot.trendHistory.map(({ model, basis, dataLoss }) => [
        'mini-batch-ema', model.step, model.revision, '', '', '', '', '', basis.latestBatchSize, dataLoss,
    ].map(String)));
    expect(await identity(page)).toEqual(later);
});

test('a saved variation stages the baseline recipe and compares evaluated results after Apply', async ({ page }) => {
    await applyDataset(page, 'XOR');
    await setupSection(page, 'Training').click();
    const learningRate = page.getByRole('textbox', { name: 'Learning rate', exact: true });
    const baselineRate = await learningRate.inputValue();
    await learning(page);
    await stepWithEvidence(page, 1);
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    await save(page, 'Baseline XOR');

    await applyDataset(page, 'Plane');
    await learning(page);
    await stepWithEvidence(page, 1);
    const previous = await identity(page);
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    const baselineActions = await openActions(page, 'Baseline XOR');
    const evidenceButton = baselineActions.getByRole('button', { name: 'Download evidence for Baseline XOR', exact: true });
    const baselineEvidence = await downloadFile(page, evidenceButton);
    await baselineActions.getByRole('button', { name: 'Try a variation', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Variation of Baseline XOR', exact: true })).toBeFocused();
    await expect(learningRate).toHaveValue(baselineRate);
    expect(await identity(page)).toEqual(previous);
    await learningRate.fill('0.031');
    expect(await identity(page)).toEqual(previous);
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(transport(page)).not.toHaveAttribute('data-model-generation', previous.generation!);
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
    await ready(page);
    await learning(page);
    await stepWithEvidence(page, 1);
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    await save(page, 'Faster XOR');
    await expect(savedRun(page, 'Baseline XOR')).toBeVisible();
    await page.getByRole('button', { name: 'Compare with baseline', exact: true }).click();
    const heading = page.getByRole('heading', { name: 'Compare experiments', exact: true });
    await expect(heading).toBeFocused();
    const comparison = page.getByRole('group', { name: /Saved run comparison:.*Faster XOR.*Baseline XOR/ });
    const changes = comparison.getByRole('region', { name: 'Recipe changes', exact: true });
    await expect(changes.getByText('Learning rate', { exact: true })).toBeVisible();
    await expect(changes).toContainText(`${baselineRate} → 0.031`);
    await expect(comparison.getByText('Not directly comparable', { exact: true })).toHaveCount(0);

    await comparison.getByRole('button', { name: '← Saved runs', exact: true }).click();
    await openActions(page, 'Baseline XOR');
    expect(await downloadFile(page, evidenceButton)).toEqual(baselineEvidence);
    await baselineActions.getByRole('button', { name: 'Try a variation', exact: true }).click();
    await applyDataset(page, 'Plane');
    await learning(page);
    await stepWithEvidence(page, 1);
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    await save(page, 'Plane variation');
    await page.getByRole('button', { name: 'Compare with baseline', exact: true }).click();
    await expect(heading).toBeFocused();
    const incomparable = page.getByRole('group', { name: /Saved run comparison:.*Plane variation.*Baseline XOR/ });
    await expect(incomparable).toContainText('Not directly comparable');
    await expect(incomparable).toContainText('Dataset and objective identities must both match before losses can be ranked.');
    await expect(incomparable.getByRole('region', { name: 'Recipe changes', exact: true })).toContainText('Dataset');
});
