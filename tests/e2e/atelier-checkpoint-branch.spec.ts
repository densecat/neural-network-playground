import { expect, test } from '@playwright/test';
import { expectEvidence, learning, ready, transport, utility } from './atelier-helpers';

test('a bounded checkpoint branch preserves the previous live model for restore', async ({ page }) => {
    await page.goto('./');
    await ready(page);
    await learning(page);
    const training = transport(page);
    await training.getByRole('button', { name: 'Run one training step', exact: true }).click();
    await expectEvidence(page, 1);

    await utility(page, 'Session checkpoints');
    const dialog = page.getByRole('dialog', { name: 'Session checkpoints', exact: true });
    const timeline = dialog.getByRole('slider', { name: 'Checkpoint timeline', exact: true });
    await timeline.press('Home');
    await expect(timeline).toHaveAttribute('aria-valuetext', 'Step 0');
    await dialog.getByLabel('Optimizer steps for this branch', { exact: true }).fill('2');
    await dialog.getByRole('button', { name: 'Branch from checkpoint', exact: true }).click();
    await expect(dialog).toContainText('Branched from Step 0. Previous live model saved at Step 1.');
    await dialog.getByRole('button', { name: 'Close Session checkpoints', exact: true }).click();
    await expectEvidence(page, 0);

    await training.getByRole('button', { name: 'Play training', exact: true }).click();
    await expect(training).toHaveAttribute('data-status', 'paused');
    await expectEvidence(page, 2);
    await expect(training.getByRole('status')).toContainText('Branch finished');

    await utility(page, 'Session checkpoints');
    await timeline.press('Home');
    await expect(timeline).toHaveAttribute('aria-valuetext', 'Step 0');
    await dialog.getByRole('button', { name: 'Select previous live checkpoint', exact: true }).click();
    await expect(timeline).toHaveAttribute('aria-valuetext', 'Step 1');
    await dialog.getByRole('button', { name: 'Restore Step 1', exact: true }).click();
    await expect(dialog).toContainText('Restored checkpoint');
    await dialog.getByRole('button', { name: 'Close Session checkpoints', exact: true }).click();
    await expectEvidence(page, 1);
    await expect(training.getByRole('button', { name: 'Play training', exact: true })).toBeEnabled();
});
