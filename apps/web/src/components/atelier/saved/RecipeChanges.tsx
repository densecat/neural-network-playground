import type { StandardExperimentRecipeV2 } from '@nn-playground/shared';
import './recipeChanges.css';

function describeOption({ kind, ...options }: { kind: string }): string {
    const parameters = Object.entries(options).sort(([left], [right]) => left.localeCompare(right));
    return [kind, ...parameters.map(([key, value]) => (
        `${key.replace(/([a-z])([A-Z0-9])/g, '$1 $2').toLowerCase()} ${String(value)}`
    ))].join(' · ');
}

function settings(recipe: StandardExperimentRecipeV2): Record<string, string> {
    return {
        'Dataset': recipe.task.dataset,
        'Samples': String(recipe.data.sampleCount),
        'Noise': `${recipe.data.noise}%`,
        'Train fraction': String(recipe.data.trainFraction),
        'Data seed': String(recipe.data.seed),
        'Input features': recipe.inputs.featureIds.join(', '),
        'Hidden layers': recipe.model.hiddenLayers.join(' → ') || 'None',
        'Hidden activation': recipe.model.hiddenActivation,
        'Weight initialization': recipe.model.initialization,
        'Model seed': String(recipe.model.seed),
        'Learning rate': String(recipe.training.learningRate),
        'Batch size': String(recipe.training.batchSize),
        'Optimizer': describeOption(recipe.training.optimizer),
        'Learning-rate schedule': describeOption(recipe.training.schedule),
        'Data loss': describeOption(recipe.objective.dataLoss),
        'Regularization': describeOption(recipe.objective.penalty),
        'Gradient clipping': describeOption(recipe.training.gradientClipping),
    };
}

export function RecipeChanges({ baseline, variation }: {
    baseline: StandardExperimentRecipeV2;
    variation: StandardExperimentRecipeV2;
}) {
    const before = settings(baseline);
    const changes = Object.entries(settings(variation)).filter(([label, value]) => before[label] !== value);
    if (changes.length === 0) return <p>No settings changed yet.</p>;
    return <dl className="atelier-recipe-changes">
        {changes.map(([label, after]) => <div key={label}>
            <dt>{label}</dt>
            <dd>{before[label]} → {after}</dd>
        </div>)}
    </dl>;
}
