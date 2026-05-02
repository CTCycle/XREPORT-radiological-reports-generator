import { useCallback, useState } from 'react';

interface MetricSelectionState<TMetric extends string> {
    selectedMetrics: TMetric[];
    isSelected: (metric: TMetric) => boolean;
    toggleMetric: (metric: TMetric) => void;
    setSelectedMetrics: (metrics: TMetric[]) => void;
}

export function useMetricSelection<TMetric extends string>(
    initialSelection: TMetric[] = []
): MetricSelectionState<TMetric> {
    const [selectedMetrics, setSelectedMetrics] = useState<TMetric[]>(initialSelection);

    const isSelected = useCallback((metric: TMetric) => {
        return selectedMetrics.includes(metric);
    }, [selectedMetrics]);

    const toggleMetric = useCallback((metric: TMetric) => {
        setSelectedMetrics((prev) => {
            if (prev.includes(metric)) {
                return prev.filter((item) => item !== metric);
            }
            return [...prev, metric];
        });
    }, []);

    return {
        selectedMetrics,
        isSelected,
        toggleMetric,
        setSelectedMetrics,
    };
}
