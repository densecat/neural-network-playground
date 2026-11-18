import { expect, test, type Page } from '@playwright/test';
import { identity, ready, transport, utility, workspace } from './atelier-helpers';

async function stepTo(page: Page, step: number): Promise<void> {
    await transport(page).getByRole('button', { name: 'Run one training step', exact: true }).click();
    await expect(transport(page)).toHaveAttribute('data-model-step', String(step));
}

async function exportParameters(page: Page, step: number): Promise<string> {
    await utility(page, 'Export / import');
    const dialog = page.getByRole('dialog', { name: 'Export / import' });
    await dialog.getByRole('tab', { name: 'Code', exact: true }).click();
    await dialog.getByRole('tab', { name: 'NumPy', exact: true }).click();
    await expect(dialog).toContainText(`Parameter snapshot: step ${step}.`);
    const code = dialog.getByRole('tabpanel', { name: 'NumPy' });
    await expect(code).toContainText('# Trained weights');
    const parameters = await code.innerText();
    await dialog.getByRole('button', { name: 'Close Export / import' }).click();
    return parameters;
}

async function predictionControls(page: Page) {
    await workspace(page, 'Results');
    await page.getByRole('tablist', { name: 'Results views' }).getByRole('tab', { name: 'Prediction', exact: true }).click();
    const controls = page.getByRole('region', { name: 'Prediction controls' });
    return [
        controls.getByRole('checkbox', { name: 'Show test data', exact: true }),
        controls.getByRole('checkbox', { name: 'Discretize output', exact: true }),
    ];
}

test.beforeEach(async ({ page }) => {
    await page.goto('./');
    await ready(page);
});

test('paused prediction display edits preserve parameters and deterministic training', async ({ page }) => {
    const controls = await predictionControls(page);
    await stepTo(page, 1);
    const firstParameters = await exportParameters(page, 1);
    const before = await identity(page);
    for (const checkbox of controls) {
        await expect(checkbox).not.toBeChecked();
        await checkbox.check();
        await expect(checkbox).toBeChecked();
        expect(await identity(page)).toMatchObject({ generation: before.generation, revision: before.revision, step: before.step });
    }
    expect(page.url()).not.toBe(before.url);
    expect(await exportParameters(page, 1)).toBe(firstParameters);
    await stepTo(page, 2);
    await stepTo(page, 3);
    const toggledParameters = await exportParameters(page, 3);

    for (const checkbox of controls) await checkbox.uncheck();
    await transport(page).getByRole('button', { name: 'Reset training', exact: true }).click();
    await expect(transport(page)).not.toHaveAttribute('data-model-generation', before.generation!);
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
    await ready(page);
    for (const step of [1, 2, 3]) await stepTo(page, step);
    expect(await exportParameters(page, 3)).toBe(toggledParameters);

    const replayed = await identity(page);
    await workspace(page, 'Setup');
    await page.getByRole('navigation', { name: 'Setup sections' }).getByRole('button', { name: 'Training', exact: true }).click();
    await page.getByLabel('Learning rate', { exact: true }).fill('0.019');
    expect(await identity(page)).toEqual(replayed);
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(transport(page)).not.toHaveAttribute('data-model-generation', replayed.generation!);
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
    await ready(page);
});

test('running and paused prediction display edits preserve the active training generation', async ({ page }) => {
    const controls = await predictionControls(page);
    const training = transport(page);
    const generation = await training.getAttribute('data-model-generation');
    const currentStep = async () => Number(await training.getAttribute('data-model-step'));
    await training.getByRole('button', { name: 'Play training', exact: true }).click();
    await expect(training).toHaveAttribute('data-status', 'running');
    await expect.poll(currentStep).toBeGreaterThan(0);
    for (const checkbox of controls) {
        const before = await currentStep();
        await checkbox.check();
        await expect(checkbox).toBeChecked();
        await expect.poll(currentStep).toBeGreaterThan(before);
        await expect(training).toHaveAttribute('data-model-generation', generation!);
        await expect(training).toHaveAttribute('data-status', 'running');
    }
    await training.getByRole('button', { name: 'Pause training', exact: true }).click();
    await expect(training).toHaveAttribute('data-status', 'paused');
    await expect(training).toHaveAttribute('data-model-generation', generation!);
    await expect(page.getByText(/Evaluation matches the current step\./)).toBeVisible();
    const paused = await identity(page);
    for (const checkbox of controls) {
        await checkbox.uncheck();
        await expect(checkbox).not.toBeChecked();
        await expect(training).toHaveAttribute('data-status', 'paused');
        expect(await identity(page)).toMatchObject({ generation, revision: paused.revision, step: paused.step });
    }
});
