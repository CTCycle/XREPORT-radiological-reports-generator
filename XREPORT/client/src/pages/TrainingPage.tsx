import { useState } from 'react';
import {
    FolderUp, FileSpreadsheet, Play, Settings,
    Database, Activity, Cpu, Sliders, BarChart2, ChevronDown, RotateCcw
} from 'lucide-react';
import './TrainingPage.css';

interface TrainingConfig {
    // Dataset Processing
    seed: number;
    sampleSize: number;
    validationSize: number;
    splitSeed: number;
    maxReportSize: number;
    tokenizer: string;

    // Dataset Evaluation
    imgStats: boolean;
    textStats: boolean;
    pixDist: boolean;

    // Model Architecture
    numEncoders: number;
    numDecoders: number;
    embeddingDims: number;
    attnHeads: number;
    freezeImgEncoder: boolean;
    trainTemp: number;

    // Model Dataset Config
    useImgAugment: boolean;
    shuffleWithBuffer: boolean;
    shuffleBufferSize: number;

    // Training Parameters
    epochs: number;
    batchSize: number;
    trainSeed: number;
    saveCheckpoints: boolean;
    checkpointFreq: number;
    mixedPrecision: boolean;
    runTensorboard: boolean;
    useJIT: boolean;
    jitBackend: string;

    // Scheduler
    useScheduler: boolean;
    targetLR: number;
    warmupSteps: number;
    realTimePlot: boolean;

    // Session
    useGpu: boolean;
    gpuId: number;
}

// Mock checkpoints for demonstration
const MOCK_CHECKPOINTS = [
    { id: 'checkpoint_epoch_50', label: 'Epoch 50 - Loss: 0.234' },
    { id: 'checkpoint_epoch_40', label: 'Epoch 40 - Loss: 0.287' },
    { id: 'checkpoint_epoch_30', label: 'Epoch 30 - Loss: 0.342' },
    { id: 'checkpoint_epoch_20', label: 'Epoch 20 - Loss: 0.456' },
    { id: 'checkpoint_epoch_10', label: 'Epoch 10 - Loss: 0.612' },
];

export default function TrainingPage() {
    const [config, setConfig] = useState<TrainingConfig>({
        // Dataset Processing
        seed: 42,
        sampleSize: 1.0,
        validationSize: 0.2,
        splitSeed: 42,
        maxReportSize: 200,
        tokenizer: 'distilbert-base-uncased',

        // Dataset Evaluation
        imgStats: false,
        textStats: false,
        pixDist: false,

        // Model Architecture
        numEncoders: 6,
        numDecoders: 6,
        embeddingDims: 768,
        attnHeads: 8,
        freezeImgEncoder: true,
        trainTemp: 1.0,

        // Model Dataset Config
        useImgAugment: false,
        shuffleWithBuffer: true,
        shuffleBufferSize: 256,

        // Training Parameters
        epochs: 100,
        batchSize: 32,
        trainSeed: 42,
        saveCheckpoints: true,
        checkpointFreq: 1,
        mixedPrecision: false,
        runTensorboard: false,
        useJIT: false,
        jitBackend: 'inductor',

        // Scheduler
        useScheduler: false,
        targetLR: 0.001,
        warmupSteps: 1000,
        realTimePlot: true,

        // Session
        useGpu: true,
        gpuId: 0
    });

    // Accordion states
    const [newSessionExpanded, setNewSessionExpanded] = useState(true);
    const [resumeSessionExpanded, setResumeSessionExpanded] = useState(false);

    // Resume training state
    const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
    const [additionalEpochs, setAdditionalEpochs] = useState(50);

    const handleConfigChange = (key: string, value: any) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    };

    const handleFolderUpload = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.webkitdirectory = true;
        input.onchange = (e) => {
            const files = (e.target as HTMLInputElement).files;
            if (files && files.length > 0) {
                console.log('Folder selected:', files);
            }
        };
        input.click();
    };

    const handleFileUpload = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.csv, .xlsx';
        input.onchange = (e) => {
            const file = (e.target as HTMLInputElement).files?.[0];
            if (file) {
                console.log('File selected:', file);
            }
        };
        input.click();
    };

    return (
        <div className="training-container">
            <div className="header">
                <h1>XREPORT Transformer</h1>
                <p>Refer to X-ray automatic reports generation</p>
            </div>

            <div className="layout-rows">
                {/* Row 1: Data Upload Only */}
                <div className="layout-row row-datasource">
                    <div className="section">
                        <div className="section-title">
                            <Database size={18} />
                            <span>Data Source</span>
                        </div>
                        <div className="upload-row-content">
                            <div className="upload-grid">
                                <div className="upload-card" onClick={handleFolderUpload}>
                                    <FolderUp className="upload-icon" />
                                    <div className="upload-text">Upload Image Folder</div>
                                    <div className="upload-subtext">Select directory</div>
                                </div>
                                <div className="upload-card" onClick={handleFileUpload}>
                                    <FileSpreadsheet className="upload-icon" />
                                    <div className="upload-text">Upload Data File</div>
                                    <div className="upload-subtext">Select .csv or .xlsx</div>
                                </div>
                            </div>
                            <button className="btn btn-secondary btn-sm">
                                Load Dataset
                            </button>
                        </div>
                    </div>
                </div>

                {/* Row 2: Dataset Processing & Evaluation */}
                <div className="layout-row row-processing-analysis">
                    {/* Processing */}
                    <div className="section">
                        <div className="section-title">
                            <Sliders size={18} />
                            <span>Dataset Processing</span>
                        </div>
                        <div className="config-grid">
                            <div className="form-group">
                                <label className="form-label">Sample Size (0-1)</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    min="0.01"
                                    max="1.0"
                                    className="form-input"
                                    value={config.sampleSize}
                                    onChange={(e) => handleConfigChange('sampleSize', parseFloat(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Seed</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={config.seed}
                                    onChange={(e) => handleConfigChange('seed', parseInt(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Val Split (0-1)</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    max="1.0"
                                    className="form-input"
                                    value={config.validationSize}
                                    onChange={(e) => handleConfigChange('validationSize', parseFloat(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Split Seed</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={config.splitSeed}
                                    onChange={(e) => handleConfigChange('splitSeed', parseInt(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Max Report Size</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={config.maxReportSize}
                                    onChange={(e) => handleConfigChange('maxReportSize', parseInt(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Tokenizer</label>
                                <select
                                    className="form-select"
                                    value={config.tokenizer}
                                    onChange={(e) => handleConfigChange('tokenizer', e.target.value)}
                                >
                                    <option value="distilbert-base-uncased">distilbert-base-uncased</option>
                                    <option value="bert-base-uncased">bert-base-uncased</option>
                                    <option value="roberta-base">roberta-base</option>
                                </select>
                            </div>
                            <div className="form-group span-3">
                                <button className="btn btn-secondary btn-sm" style={{ marginTop: '0.25rem' }}>
                                    Build Dataset
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Evaluation */}
                    <div className="section">
                        <div className="section-title">
                            <BarChart2 size={18} />
                            <span>Dataset Evaluation</span>
                        </div>
                        <div className="config-grid" style={{ gridTemplateColumns: '1fr' }}>
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.imgStats}
                                    onChange={(e) => handleConfigChange('imgStats', e.target.checked)}
                                />
                                <div className="checkbox-visual" />
                                <span className="checkbox-label">Image statistics</span>
                            </label>
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.textStats}
                                    onChange={(e) => handleConfigChange('textStats', e.target.checked)}
                                />
                                <div className="checkbox-visual" />
                                <span className="checkbox-label">Text statistics</span>
                            </label>
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.pixDist}
                                    onChange={(e) => handleConfigChange('pixDist', e.target.checked)}
                                />
                                <div className="checkbox-visual" />
                                <span className="checkbox-label">Pixel intensity dist.</span>
                            </label>
                            <button className="btn btn-secondary btn-sm" style={{ marginTop: '0.25rem' }}>
                                View Evaluation
                            </button>
                        </div>
                    </div>
                </div>

                {/* Row 3: Training Session Accordions */}
                <div className="accordion-row">
                    {/* New Training Session Accordion */}
                    <div className="accordion">
                        <div
                            className="accordion-header"
                            onClick={() => setNewSessionExpanded(!newSessionExpanded)}
                        >
                            <div className="accordion-header-left">
                                <Play size={18} />
                                <span>New Training Session</span>
                            </div>
                            <ChevronDown
                                size={20}
                                className={`accordion-chevron ${newSessionExpanded ? 'expanded' : ''}`}
                            />
                        </div>
                        {newSessionExpanded && <div className="accordion-divider" />}
                        <div className={`accordion-content ${newSessionExpanded ? 'expanded' : ''}`}>
                            <div className="accordion-content-inner">
                                <div className="row-training-controls">
                                    {/* Left Column: Model Architecture & Dataset Config */}
                                    <div className="section-column">
                                        <div className="section">
                                            <div className="section-title">
                                                <Cpu size={18} />
                                                <span>Model Architecture</span>
                                            </div>
                                            <div className="config-grid">
                                                <div className="form-group">
                                                    <label className="form-label">Encoders</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={config.numEncoders}
                                                        onChange={(e) => handleConfigChange('numEncoders', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Decoders</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={config.numDecoders}
                                                        onChange={(e) => handleConfigChange('numDecoders', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Embed Dims</label>
                                                    <input
                                                        type="number"
                                                        step="8"
                                                        className="form-input"
                                                        value={config.embeddingDims}
                                                        onChange={(e) => handleConfigChange('embeddingDims', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Attn Heads</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={config.attnHeads}
                                                        onChange={(e) => handleConfigChange('attnHeads', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Temp</label>
                                                    <input
                                                        type="number"
                                                        step="0.05"
                                                        className="form-input"
                                                        value={config.trainTemp}
                                                        onChange={(e) => handleConfigChange('trainTemp', parseFloat(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group" style={{ alignSelf: 'end' }}>
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.freezeImgEncoder}
                                                            onChange={(e) => handleConfigChange('freezeImgEncoder', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Freeze Encoder</span>
                                                    </label>
                                                </div>
                                            </div>

                                            <div className="sub-section-title">
                                                <Settings size={16} />
                                                <span>Dataset Config</span>
                                            </div>
                                            <div className="config-grid">
                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.useImgAugment}
                                                            onChange={(e) => handleConfigChange('useImgAugment', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Image Augment</span>
                                                    </label>
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.shuffleWithBuffer}
                                                            onChange={(e) => handleConfigChange('shuffleWithBuffer', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Shuffle Buffer</span>
                                                    </label>
                                                </div>
                                                {config.shuffleWithBuffer && (
                                                    <div className="form-group">
                                                        <label className="form-label">Buffer Size</label>
                                                        <input
                                                            type="number"
                                                            step="10"
                                                            className="form-input"
                                                            value={config.shuffleBufferSize}
                                                            onChange={(e) => handleConfigChange('shuffleBufferSize', parseInt(e.target.value))}
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Right Column: Training Parameters */}
                                    <div className="section-column">
                                        <div className="section">
                                            <div className="section-title">
                                                <Activity size={18} />
                                                <span>Training Parameters</span>
                                            </div>
                                            <div className="config-grid">
                                                <div className="form-group">
                                                    <label className="form-label">Epochs</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={config.epochs}
                                                        onChange={(e) => handleConfigChange('epochs', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Batch Size</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={config.batchSize}
                                                        onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Training Seed</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={config.trainSeed}
                                                        onChange={(e) => handleConfigChange('trainSeed', parseInt(e.target.value))}
                                                    />
                                                </div>

                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.saveCheckpoints}
                                                            onChange={(e) => handleConfigChange('saveCheckpoints', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Checkpoints</span>
                                                    </label>
                                                </div>

                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.runTensorboard}
                                                            onChange={(e) => handleConfigChange('runTensorboard', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Tensorboard</span>
                                                    </label>
                                                </div>

                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.mixedPrecision}
                                                            onChange={(e) => handleConfigChange('mixedPrecision', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Mixed Prec.</span>
                                                    </label>
                                                </div>

                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.useJIT}
                                                            onChange={(e) => handleConfigChange('useJIT', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">JIT Compiler</span>
                                                    </label>
                                                </div>

                                                {config.useJIT && (
                                                    <div className="form-group span-2">
                                                        <label className="form-label">JIT Backend</label>
                                                        <select
                                                            className="form-select"
                                                            value={config.jitBackend}
                                                            onChange={(e) => handleConfigChange('jitBackend', e.target.value)}
                                                        >
                                                            <option value="inductor">inductor</option>
                                                            <option value="eager">eager</option>
                                                            <option value="aot_eager">aot_eager</option>
                                                            <option value="nvprims_nvfuser">nvprims_nvfuser</option>
                                                        </select>
                                                    </div>
                                                )}

                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.useScheduler}
                                                            onChange={(e) => handleConfigChange('useScheduler', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">LR Scheduler</span>
                                                    </label>
                                                </div>

                                                {config.useScheduler && (
                                                    <>
                                                        <div className="form-group">
                                                            <label className="form-label">Target LR</label>
                                                            <input
                                                                type="number"
                                                                step="0.0001"
                                                                className="form-input"
                                                                value={config.targetLR}
                                                                onChange={(e) => handleConfigChange('targetLR', parseFloat(e.target.value))}
                                                            />
                                                        </div>
                                                        <div className="form-group">
                                                            <label className="form-label">Warmup Steps</label>
                                                            <input
                                                                type="number"
                                                                className="form-input"
                                                                value={config.warmupSteps}
                                                                onChange={(e) => handleConfigChange('warmupSteps', parseInt(e.target.value))}
                                                            />
                                                        </div>
                                                    </>
                                                )}

                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={config.realTimePlot}
                                                            onChange={(e) => handleConfigChange('realTimePlot', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Real-time Plot</span>
                                                    </label>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Actions Bar inside accordion */}
                                <div className="actions-bar" style={{ marginTop: '1rem' }}>
                                    <button className="btn btn-primary">
                                        <Play size={16} />
                                        Start Training
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Resume Training Session Accordion */}
                    <div className="accordion">
                        <div
                            className="accordion-header"
                            onClick={() => setResumeSessionExpanded(!resumeSessionExpanded)}
                        >
                            <div className="accordion-header-left">
                                <RotateCcw size={18} />
                                <span>Resume Training Session</span>
                            </div>
                            <ChevronDown
                                size={20}
                                className={`accordion-chevron ${resumeSessionExpanded ? 'expanded' : ''}`}
                            />
                        </div>
                        {resumeSessionExpanded && <div className="accordion-divider" />}
                        <div className={`accordion-content ${resumeSessionExpanded ? 'expanded' : ''}`}>
                            <div className="accordion-content-inner">
                                <div className="resume-session-grid">
                                    <div className="form-group">
                                        <label className="form-label">Select Checkpoint</label>
                                        <select
                                            className="form-select"
                                            value={selectedCheckpoint}
                                            onChange={(e) => setSelectedCheckpoint(e.target.value)}
                                        >
                                            <option value="">-- Select a checkpoint --</option>
                                            {MOCK_CHECKPOINTS.map((cp) => (
                                                <option key={cp.id} value={cp.id}>
                                                    {cp.label}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Additional Epochs</label>
                                        <input
                                            type="number"
                                            className="form-input"
                                            value={additionalEpochs}
                                            onChange={(e) => setAdditionalEpochs(parseInt(e.target.value))}
                                            min={1}
                                        />
                                    </div>
                                    <button
                                        className="btn btn-primary"
                                        disabled={!selectedCheckpoint}
                                    >
                                        <RotateCcw size={16} />
                                        Resume Training
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
