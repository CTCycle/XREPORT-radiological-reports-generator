import { BarChart2, FileText, Image, Loader, CheckCircle, AlertCircle } from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';
import './ValidationDashboard.css';
import {
    ValidationResponse,
    PixelDistribution,
    ImageStatistics,
    TextStatistics,
} from '../services/validationService';

interface ValidationDashboardProps {
    isLoading: boolean;
    validationResult: ValidationResponse | null;
    error: string | null;
}

function PixelDistributionChart({ data }: { data: PixelDistribution }) {
    // Downsample to 64 bins for better visualization
    const binSize = 4;
    const chartData: { bin: number; count: number }[] = [];
    for (let i = 0; i < 256; i += binSize) {
        const count = data.counts
            .slice(i, i + binSize)
            .reduce((sum, val) => sum + val, 0);
        chartData.push({ bin: i, count });
    }

    return (
        <div className="chart-section">
            <div className="chart-title">
                <BarChart2 size={16} style={{ display: 'inline', marginRight: '8px' }} />
                Pixel Intensity Distribution
            </div>
            <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis
                        dataKey="bin"
                        stroke="#9ca3af"
                        tick={{ fill: '#9ca3af', fontSize: 11 }}
                        label={{ value: 'Pixel Intensity', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
                    />
                    <YAxis
                        stroke="#9ca3af"
                        tick={{ fill: '#9ca3af', fontSize: 11 }}
                        tickFormatter={(value) => {
                            if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
                            if (value >= 1000) return `${(value / 1000).toFixed(0)}K`;
                            return value.toString();
                        }}
                    />
                    <Tooltip
                        contentStyle={{
                            background: 'rgba(30, 30, 35, 0.95)',
                            border: '1px solid rgba(255, 215, 0, 0.2)',
                            borderRadius: '8px',
                        }}
                        labelFormatter={(value) => `Intensity: ${value}-${Number(value) + binSize - 1}`}
                        formatter={(value) => [typeof value === 'number' ? value.toLocaleString() : String(value ?? 0), 'Count']}
                    />
                    <Bar dataKey="count" fill="#ffd700" opacity={0.8} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}

function TextStatsWidget({ data }: { data: TextStatistics }) {
    return (
        <div className="stats-section">
            <div className="stats-section-title">
                <FileText size={16} />
                Text Statistics
            </div>
            <div className="stats-row">
                <span className="stat-label">Total Reports</span>
                <span className="stat-value highlight">{data.count.toLocaleString()}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Total Words</span>
                <span className="stat-value">{data.total_words.toLocaleString()}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Unique Words (Vocabulary)</span>
                <span className="stat-value">{data.unique_words.toLocaleString()}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Avg Words/Report</span>
                <span className="stat-value">{data.avg_words_per_report.toFixed(1)}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Min Words/Report</span>
                <span className="stat-value">{data.min_words_per_report}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Max Words/Report</span>
                <span className="stat-value">{data.max_words_per_report}</span>
            </div>
        </div>
    );
}

function ImageStatsWidget({ data }: { data: ImageStatistics }) {
    return (
        <div className="stats-section">
            <div className="stats-section-title">
                <Image size={16} />
                Image Statistics
            </div>
            <div className="stats-row">
                <span className="stat-label">Total Images</span>
                <span className="stat-value highlight">{data.count.toLocaleString()}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Avg Height</span>
                <span className="stat-value">{data.mean_height.toFixed(0)} px</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Avg Width</span>
                <span className="stat-value">{data.mean_width.toFixed(0)} px</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Avg Pixel Value</span>
                <span className="stat-value">{data.mean_pixel_value.toFixed(2)}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Std Pixel Value</span>
                <span className="stat-value">{data.std_pixel_value.toFixed(2)}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Avg Noise Std</span>
                <span className="stat-value">{data.mean_noise_std.toFixed(2)}</span>
            </div>
            <div className="stats-row">
                <span className="stat-label">Avg Noise Ratio</span>
                <span className="stat-value">{data.mean_noise_ratio.toFixed(4)}</span>
            </div>
        </div>
    );
}

export default function ValidationDashboard({
    isLoading,
    validationResult,
    error,
}: ValidationDashboardProps) {
    const hasResults = validationResult?.success && (
        validationResult.text_statistics ||
        validationResult.image_statistics ||
        validationResult.pixel_distribution
    );

    return (
        <div className="validation-dashboard">
            <div className="dashboard-header">
                <div className="dashboard-title">
                    <BarChart2 size={20} />
                    Dataset Validation Results
                </div>
                {hasResults && (
                    <div className="dashboard-status success">
                        <CheckCircle size={14} />
                        Complete
                    </div>
                )}
                {error && (
                    <div className="dashboard-status error">
                        <AlertCircle size={14} />
                        Error
                    </div>
                )}
            </div>

            {isLoading ? (
                <div className="loading-container">
                    <Loader size={32} className="spin" />
                    <span className="loading-text">Running validation analytics...</span>
                </div>
            ) : error ? (
                <div className="idle-message" style={{ color: '#ef4444' }}>
                    {error}
                </div>
            ) : hasResults ? (
                <div className="validation-content">
                    {/* Stats Grid */}
                    <div className="stats-grid">
                        {validationResult.text_statistics && (
                            <TextStatsWidget data={validationResult.text_statistics} />
                        )}
                        {validationResult.image_statistics && (
                            <ImageStatsWidget data={validationResult.image_statistics} />
                        )}
                    </div>

                    {/* Pixel Distribution Chart */}
                    {validationResult.pixel_distribution && (
                        <PixelDistributionChart data={validationResult.pixel_distribution} />
                    )}
                </div>
            ) : (
                <div className="idle-message">
                    Select validation options and click "View Evaluation" to see results.
                </div>
            )}
        </div>
    );
}
