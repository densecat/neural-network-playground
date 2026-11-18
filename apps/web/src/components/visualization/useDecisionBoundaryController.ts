import { useCallback, useEffect, useMemo, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';
import { useDecisionBoundaryModel } from './useDecisionBoundaryModel.ts';
import { getDecisionOverlayCopy, type DecisionOverlayMode } from './decisionBoundaryModel.ts';

export function useDecisionOverlayPreference() {
    const taskKind = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe.task.kind : null);
    const [overlayMode, setOverlayMode] = useState<DecisionOverlayMode>('none');
    useEffect(() => { setOverlayMode('none'); }, [taskKind]);
    return useMemo(() => ({ overlayMode, setOverlayMode }), [overlayMode]);
}

export type DecisionOverlayPreference = ReturnType<typeof useDecisionOverlayPreference>;

/** One frame snapshot shared by the plot and its evidence controls. */
export function useDecisionBoundaryController(preference?: DecisionOverlayPreference) {
    const showTestData = usePlaygroundStore((s) => s.access.status === 'ready' && s.access.prepared.document.view.showTestData);
    const discretize = usePlaygroundStore((s) => s.access.status === 'ready' && s.access.prepared.document.view.discretizeOutput);
    const showHelp = usePlaygroundStore((s) => s.access.status === 'ready' && s.access.prepared.document.recipe.task.kind !== 'regression');
    const editView = usePlaygroundStore((s) => s.editView);
    const trainPoints = useTrainingStore((s) => s.trainPoints);
    const testPoints = useTrainingStore((s) => s.testPoints);
    const taskKind = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe.task.kind : null);
    const localPreference = useDecisionOverlayPreference();
    const { overlayMode, setOverlayMode } = preference ?? localPreference;
    const guidanceLevel = useAudienceGuidanceLevel();
    const model = useDecisionBoundaryModel({ trainPoints, testPoints, showTestData, discretize, overlayMode });
    const setShowTestData = useCallback((checked: boolean) => editView((view) => ({ ...view, showTestData: checked })), [editView]);
    const setDiscretize = useCallback((checked: boolean) => editView((view) => ({ ...view, discretizeOutput: checked })), [editView]);
    const commands = useMemo(() => ({ setShowTestData, setDiscretize, setOverlayMode }), [setShowTestData, setDiscretize, setOverlayMode]);
    return { model, taskKind, showTestData, discretize, showHelp, guidanceLevel, overlayMode, overlayCopy: getDecisionOverlayCopy(overlayMode, showTestData, discretize), commands };
}
export type DecisionBoundaryController = ReturnType<typeof useDecisionBoundaryController>;
