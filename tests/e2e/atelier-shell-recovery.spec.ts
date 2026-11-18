import { expect, test } from '@playwright/test';
import { identity, ready, transport, workspace } from './atelier-helpers';

type CommitCounts = { shell: number; graph: number; transport: number };
type ProfileWindow = Window & { atelierCommitCounts: CommitCounts };

test('a rejected Saved runs chunk leaves navigation and the live model available', async ({ page }) => {
    let failures = 0;
    await page.route(/\/WorkspaceUtilities-[^/]+\.js$/, (route) => {
        failures++;
        return route.abort('failed');
    });
    await page.goto('./');
    await ready(page);
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expect(transport(page)).toHaveAttribute('data-model-step', '1');
    const before = await identity(page);

    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Saved runs unavailable' })).toBeVisible();
    await expect(page.getByText(/The code for this view could not be loaded/)).toBeVisible();
    expect(failures).toBeGreaterThan(0);
    await expect(page.getByRole('button', { name: 'Try again', exact: true })).toHaveCount(0);

    await workspace(page, 'Network');
    await expect(page.getByRole('heading', { name: 'Network', exact: true })).toBeVisible();
    expect(await identity(page)).toEqual(before);
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expect(transport(page)).toHaveAttribute('data-model-step', '2');
    await expect(transport(page)).toHaveAttribute('data-model-generation', before.generation!);
});

test('slow worker startup reserves plot space while stopped controls wait for the dataset', async ({ page }) => {
    let release!: () => void;
    const waitForRelease = new Promise<void>((resolve) => { release = resolve; });
    const worker = /\/training\.worker-[^/]+\.js$/;
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.route(worker, async (route) => {
        await waitForRelease;
        await route.continue();
    });
    try {
        await page.goto('./', { waitUntil: 'domcontentloaded' });
        await expect(page.getByRole('button', { name: 'Play training', exact: true })).toBeDisabled();
        await expect(transport(page)).toContainText('Preparing the model and dataset…');
        const dataPlot = page.locator('.atelier-network-data .atelier-data-plot');
        const predictionPlot = page.locator('.atelier-network-prediction .decision-boundary');
        await expect(dataPlot).toHaveText('Loading dataset…');
        await expect(predictionPlot).toHaveText('Loading dataset…');
        const reserved = await Promise.all([dataPlot.boundingBox(), predictionPlot.boundingBox()]);
        expect(reserved.every((box) => box && box.width > 100 && box.height > 100)).toBe(true);
        release();
        await ready(page);
        await expect(transport(page)).toContainText('Ready · choose Play to begin');
        await expect(predictionPlot.locator('canvas')).toBeVisible();
        const plotted = await Promise.all([dataPlot.boundingBox(), predictionPlot.boundingBox()]);
        for (const [index, box] of plotted.entries()) {
            expect(Math.abs(box!.height - reserved[index]!.height)).toBeLessThanOrEqual(1);
        }
    } finally {
        release();
    }
});

test('tablet Network leads in the visual and keyboard order before the compact region switch', async ({ page }) => {
    await page.setViewportSize({ width: 1199, height: 900 });
    await page.goto('./');
    await ready(page);
    const invitation = page.getByRole('button', { name: 'Dismiss lesson suggestion' });
    if (await invitation.isVisible()) await invitation.click();
    const panels = page.locator('.atelier-network > section');
    const primaryTab = page.getByRole('tablist', { name: 'Experiment workspace' }).getByRole('tab', { name: 'Network', exact: true });
    const editLayers = page.getByRole('button', { name: 'Edit layers' });
    for (const width of [1199, 1024, 768]) {
        await page.setViewportSize({ width, height: 900 });
        await expect(page.locator('.atelier-network')).toHaveAttribute('data-layout', 'stacked');
        await expect(panels.locator('h2')).toHaveText(['Network', 'Data', 'Prediction']);
        const [model, data, prediction] = await Promise.all((await panels.all()).map((panel) => panel.boundingBox()));
        expect(model!.y + model!.height).toBeLessThan(data!.y);
        expect(data!.y).toBe(prediction!.y);
        await primaryTab.focus();
        await page.keyboard.press('Tab');
        await expect(editLayers).toBeFocused();
    }
    await page.setViewportSize({ width: 740, height: 900 });
    const regions = page.getByRole('tablist', { name: 'Network regions' });
    await expect(regions.getByRole('tab')).toHaveText(['Network', 'Data', 'Prediction']);
    await expect(regions.getByRole('tab', { name: 'Network' })).toHaveAttribute('aria-selected', 'true');
    await expect(editLayers).toBeFocused();
    await expect(page.getByRole('heading', { name: 'Data', exact: true })).toBeHidden();
    await regions.getByRole('tab', { name: 'Network' }).focus();
    await page.keyboard.press('ArrowRight');
    await expect(regions.getByRole('tab', { name: 'Data' })).toBeFocused();
    await expect(page.getByRole('heading', { name: 'Data', exact: true })).toBeVisible();
});

test('Play retains its visible name and Save run opens the capture field with keyboard focus', async ({ page }) => {
    await page.goto('./');
    await ready(page);
    const play = page.getByRole('button', { name: 'Play training', exact: true });
    await expect(play).toHaveText('Play');
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expect(transport(page)).toHaveAttribute('data-model-step', '1');
    await expect(play).toBeEnabled();
    const before = await identity(page);
    await page.getByRole('button', { name: 'Save run…', exact: true }).press('Enter');
    const name = page.getByRole('textbox', { name: 'Run name' });
    await expect(name).toBeFocused();
    await expect(page.getByRole('button', { name: 'Save current run', exact: true })).toBeVisible();
    expect(await identity(page)).toEqual(before);
});

test('seeded training commits the graph only on Network and keeps the shell idle', async ({ page }, info) => {
    await page.addInitScript(() => {
        type Fiber = { stateNode?: unknown; child: Fiber | null; sibling: Fiber | null; return: Fiber | null; flags: number };
        const measuredWindow = window as ProfileWindow & { __REACT_DEVTOOLS_GLOBAL_HOOK__?: unknown };
        const previous: Partial<Record<keyof CommitCounts, Fiber>> = {};
        const targets = { shell: 'atelier', graph: 'network-graph-frame', transport: 'atelier-transport' } as const;
        measuredWindow.atelierCommitCounts = { shell: 0, graph: 0, transport: 0 };
        // React 19 marks a component that rendered with the PerformedWork flag.
        // Read fresh profiler commits through the DevTools hook in this test only.
        measuredWindow.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
            supportsFiber: true,
            inject: () => 1,
            onCommitFiberRoot: (_id: number, root: { current: Fiber }) => {
                const visit = (node: Fiber | null) => {
                    for (let fiber = node; fiber; fiber = fiber.sibling) {
                        if (fiber.stateNode instanceof HTMLElement && fiber.return) {
                            const label = (Object.keys(targets) as (keyof CommitCounts)[]).find((key) => (fiber!.stateNode as HTMLElement).classList.contains(targets[key]));
                            if (label) {
                                const component = fiber.return;
                                if (component !== previous[label] && (component.flags & 1)) measuredWindow.atelierCommitCounts[label]++;
                                previous[label] = component;
                            }
                        }
                        if (fiber.child) visit(fiber.child);
                    }
                };
                visit(root.current);
            },
        };
    });
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('./');
    await ready(page);
    await page.getByRole('combobox', { name: 'Steps per frame' }).selectOption('1');
    await page.getByRole('button', { name: 'Play training', exact: true }).click();
    const sample = async () => {
        await page.evaluate(() => { (window as ProfileWindow).atelierCommitCounts = { shell: 0, graph: 0, transport: 0 }; });
        const step = Number(await transport(page).getAttribute('data-model-step'));
        await expect.poll(async () => Number(await transport(page).getAttribute('data-model-step'))).toBeGreaterThan(step + 15);
        return page.evaluate(() => (window as ProfileWindow).atelierCommitCounts);
    };
    const network = await sample();
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Save current run', exact: true })).toBeVisible();
    const savedRuns = await sample();

    await info.attach('react-commit-counts', { body: JSON.stringify({ network, savedRuns }, null, 2), contentType: 'application/json' });
    expect(network.graph).toBeGreaterThan(0);
    expect(network.shell).toBe(0);
    expect(savedRuns.graph).toBe(0);
    expect(savedRuns.shell).toBe(0);
    expect(savedRuns.transport).toBeGreaterThan(0);
});
