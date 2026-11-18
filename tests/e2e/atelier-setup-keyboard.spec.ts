import process from 'node:process';
import { expect, test } from '@playwright/test';
import { identity, ready, transport } from './atelier-helpers';

for (const zoom of [1, 2]) {
    test(`Setup fields and Apply stay reachable by keyboard at ${zoom * 100}% document zoom`, async ({ page, browserName }) => {
        await page.setViewportSize({ width: 1024, height: 650 });
        await page.goto('./');
        await ready(page);
        await page.evaluate(async (value) => {
            document.documentElement.style.zoom = String(value);
            await document.fonts.ready;
        }, zoom);

        const tab = browserName === 'webkit' && process.platform === 'darwin' ? 'Alt+Tab' : 'Tab';
        const tabs = page.getByRole('tablist', { name: 'Experiment workspace' });
        await tabs.getByRole('tab', { name: 'Network', exact: true }).focus();
        await page.keyboard.press('Home');
        const setupTab = tabs.getByRole('tab', { name: 'Setup', exact: true });
        await expect(setupTab).toBeFocused();
        await expect(setupTab).toHaveAttribute('aria-selected', 'true');

        const sections = page.getByRole('navigation', { name: 'Setup sections' });
        for (const name of ['Data', 'Inputs & layers', 'Training']) {
            await page.keyboard.press(tab);
            await expect(sections.getByRole('button', { name, exact: true })).toBeFocused();
        }
        await page.keyboard.press('Enter');
        const setup = page.getByRole('region', { name: 'Experiment setup' });
        await expect(sections.getByRole('button', { name: 'Training', exact: true })).toHaveAttribute('aria-current', 'page');
        await expect(setup.locator('.atelier-setup-fields')).toHaveCSS('overflow-y', 'visible');
        expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);

        const before = await identity(page);
        const initialScroll = await page.evaluate(() => window.scrollY);
        await page.keyboard.press(tab);
        await page.keyboard.press(tab);
        const learningRate = setup.getByRole('textbox', { name: 'Learning rate', exact: true });
        await expect(learningRate).toBeFocused();
        await page.keyboard.press('ControlOrMeta+A');
        await page.keyboard.insertText('0.019');
        await expect(learningRate).toHaveValue('0.019');
        expect(await identity(page)).toEqual(before);

        const clipping = setup.getByRole('combobox', { name: 'Clipping', exact: true });
        for (let attempt = 0; attempt < 12 && !await clipping.evaluate((node) => node === document.activeElement); attempt++) {
            await page.keyboard.press(tab);
        }
        await expect(clipping).toBeFocused();
        await page.keyboard.press(tab);
        await expect(setup.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused();
        await page.keyboard.press(tab);
        const apply = setup.getByRole('button', { name: 'Apply changes', exact: true });
        await expect(apply).toBeFocused();
        await expect(apply).toBeInViewport();
        await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(initialScroll);
        await page.keyboard.press('Enter');
        await expect(transport(page)).not.toHaveAttribute('data-model-generation', before.generation!);
        await expect(setup).toHaveAttribute('aria-busy', 'false');
        await expect(learningRate).toHaveValue('0.019');
        await expect(apply).toBeDisabled();
    });
}
