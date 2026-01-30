import '../pages/TrainingPage.css';

interface WizardStepsProps {
    steps: string[];
    current: number;
}

export default function WizardSteps({ steps, current }: WizardStepsProps) {
    return (
        <div className="wizard-steps">
            {steps.map((step, index) => {
                const isActive = index === current;
                const isComplete = index < current;
                return (
                    <div className="wizard-step" key={step}>
                        <span className={`wizard-step-dot ${isActive || isComplete ? 'active' : ''}`}>
                            {index + 1}
                        </span>
                        <span className={`wizard-step-label ${isActive ? 'active' : ''}`}>{step}</span>
                        {index < steps.length - 1 && (
                            <span className={`wizard-step-line ${isComplete ? 'active' : ''}`} />
                        )}
                    </div>
                );
            })}
        </div>
    );
}
