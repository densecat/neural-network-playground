import { memo, useMemo, type CSSProperties } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { NetworkGraph } from '../visualization/NetworkGraph.tsx';
import { NetworkSelectionDeck } from '../visualization/NetworkSelectionDeck.tsx';
import type { DecisionOverlayPreference } from '../visualization/useDecisionBoundaryController.ts';
import { useNetworkSelectionModel, type NetworkSelectionState } from '../visualization/useNetworkSelectionController.ts';
import type { SetupTab } from '../../productShell/atelierTypes.ts';
import { Icon, Tabs } from './ui.tsx';
import { getDatasetContract } from '@nn-playground/engine';
import { useThemeStore, readPlotPalette } from '../../store/theme.ts';
import { CLASS_COLORS, fieldColor, hexRgb } from '../visualization/plotColors.ts';
import { AtelierPrediction, DatasetPlotStatus, selectDatasetPlotStatus } from './AtelierPrediction.tsx';

const DataPlot = memo(function DataPlot() {
    const status = useTrainingStore(selectDatasetPlotStatus);
    const points = useTrainingStore((s) => s.trainPoints);
    const test = useTrainingStore((s) => s.testPoints);
    const task = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe.task : null);
    const noise = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe.data.noise : 0);
    const theme = useThemeStore((s) => s.resolved);
    const background = useMemo(() => { void theme; return hexRgb(readPlotPalette().background); }, [theme]);
    const regression = task?.kind === 'regression';
    const range = useMemo(() => {
        const domain = task && getDatasetContract(task.dataset).targetDomain;
        return domain?.kind === 'continuous' ? domain.boundsForNoise(noise) : [0,1];
    }, [task,noise]);
    if (status !== 'ready') return <DatasetPlotStatus className="atelier-data-plot" status={status} />;
    return <>
        <svg className="atelier-data-plot" viewBox="0 0 256 256" role="img" aria-label={`${points.length} training samples plotted by x and y; ${regression ? 'color shows continuous target' : 'color shows target class'}`}>
            {[64,128,192].map((p) => <g key={p} stroke="var(--border-color)" strokeWidth=".5"><path d={`M${p} 0V256 M0 ${p}H256`} /></g>)}
            {points.map((point, index) => <circle key={index} cx={16+(point.x+1)*112} cy={16+(1-point.y)*112} r={2.3} fill={regression ? `rgb(${fieldColor((point.label-range[0])/(range[1]-range[0] || 1),background).join(',')})` : CLASS_COLORS[point.label] ?? CLASS_COLORS[0]} stroke="var(--bg-primary)" strokeWidth=".5" />)}
            <text x="240" y="250" fill="var(--text-secondary)" fontSize="14">x</text><text x="6" y="12" fill="var(--text-secondary)" fontSize="14">y</text>
        </svg>
        <p className="atelier-plot-caption">{points.length} training · {test.length} test samples<br />Active experiment · deterministic data split</p>
        {regression ? <p className="atelier-plot-caption">Target range {Number.isFinite(range[0]) ? range[0].toFixed(2) : '—'} to {Number.isFinite(range[1]) ? range[1].toFixed(2) : '—'}</p> : <div className="atelier-class-legend">{CLASS_COLORS.slice(0,task?.kind === 'multiclass-classification' ? 3 : 2).map((color,index) => <span key={index}><i style={{ '--class-color':color } as CSSProperties} />Class {index}</span>)}</div>}
    </>;
});

export type NetworkRegion = 'data' | 'network' | 'prediction';
interface AtelierNetworkProps {
    layout: 'compact' | 'stacked' | 'wide';
    region: NetworkRegion;
    onRegionChange(region: NetworkRegion): void;
    overlay: DecisionOverlayPreference;
    selection: NetworkSelectionState;
    onSetup(tab: SetupTab): void;
    onResults(): void;
}

export function AtelierNetwork({ layout, region, onRegionChange, overlay, selection, onSetup, onResults }: AtelierNetworkProps) {
    const compact = layout === 'compact';
    const showData = !compact || region === 'data';
    const showNetwork = !compact || region === 'network';
    const showPrediction = !compact || region === 'prediction';
    const graph = useNetworkSelectionModel(selection, showNetwork);
    const dataPanel = <section key="data" className="atelier-network-data" hidden={!showData}>
        <div className="atelier-section-title"><h2>Data</h2><button type="button" onClick={() => onSetup('dataset')} aria-label="Edit dataset"><Icon name="settings" /></button></div>
        {showData && <DataPlot />}
    </section>;
    const networkPanel = <section key="network" className="atelier-network-model" hidden={!showNetwork}>
        <div className="atelier-section-title"><h2>Network</h2><button type="button" onClick={() => onSetup('network')}>Edit layers <Icon name="next" size={14} /></button></div>
        <div className="atelier-network-graph">{showNetwork && <NetworkGraph selectionController={graph} />}</div>
    </section>;
    return <>
        {compact && <Tabs label="Network regions" items={[{id:'network',label:'Network'},{id:'data',label:'Data'},{id:'prediction',label:'Prediction'}]} value={region} onChange={onRegionChange} />}
        <div className="atelier-network" data-layout={layout}>
            {layout === 'wide' ? [dataPanel, networkPanel] : [networkPanel, dataPanel]}
            <section className="atelier-network-prediction" hidden={!showPrediction}>
                <div className="atelier-section-title"><h2>Prediction</h2><button type="button" onClick={onResults} aria-label="Expand prediction results"><Icon name="expand" /></button></div>
                {showPrediction && <AtelierPrediction overlay={overlay} />}
            </section>
        </div>
        {graph.model.kind === 'selected' && <div className="atelier-neuron-detail"><NetworkSelectionDeck model={graph.model} onClear={graph.commands.clearSelection} /></div>}
    </>;
}
