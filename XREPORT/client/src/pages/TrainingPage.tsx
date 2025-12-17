import { useState } from 'react';
import { FolderUp, FileSpreadsheet, Play, Save, Settings, Database, Activity, Cpu } from 'lucide-react';
import './TrainingPage.css';

interface TrainingConfig {
    seed: number;
    sampleSize: number;
    validationSize: number;
    splitSeed: number;
    tokenizer: string;
    epochs: number;
    batchSize: number;
    trainSeed: number;
    saveCheckpoints: boolean;
    checkpointFreq: number;
    numEncoders: number;
    numDecoders: number;
    embeddingDims: number;
    attnHeads: number;
    useGpu: boolean;
    gpuId: number;
    mixedPrecision: boolean;
    runTensorboard: boolean;
    [key: string]: any;
}

export default function TrainingPage() {
    const [config, setConfig] = useState<TrainingConfig>({
        // Dataset
        seed: 42,
        sampleSize: 100,
        validationSize: 0.2,
        splitSeed: 42,
        tokenizer: 'distilbert-base-uncased',

        // Training
        epochs: 10,
        batchSize: 32,
        trainSeed: 42,
        saveCheckpoints: true,
        checkpointFreq: 1,

        // Model
        numEncoders: 6,
        numDecoders: 6,
        embeddingDims: 768,
        attnHeads: 8,

        // Session
        useGpu: true,
        gpuId: 0,
        mixedPrecision: true,
        runTensorboard: false
    });

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
                // Here we would handle the folder upload logic
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
                // Here we would handle the file upload logic
            }
        };
        input.click();
    };

    return (
        <div className="training-container">
            <div className="header">
                <h1>Training Configuration</h1>
                <p>Manage datasets, configure model parameters, and execute training sessions.</p>
            </div>

            {/* Upload Section */}
            <div className="section">
                <div className="section-title">
                    <Database size={20} />
                    <span>Data Source</span>
                </div>
                <div className="upload-grid">
                    <div className="upload-card" onClick={handleFolderUpload}>
                        <FolderUp className="upload-icon" />
                        <div>
                            <div className="upload-text">Upload Image Folder</div>
                            <div className="upload-subtext">Select a directory containing images</div>
                        </div>
                    </div>

                    <div className="upload-card" onClick={handleFileUpload}>
                        <FileSpreadsheet className="upload-icon" />
                        <div>
                            <div className="upload-text">Upload Data File</div>
                            <div className="upload-subtext">Select .csv or .xlsx file</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Configuration Form */}
            <div className="config-grid-container">
                {/* Dataset Settings */}
                <div className="section">
                    <div className="section-title">
                        <Settings size={20} />
                        <span>Dataset Settings</span>
                    </div>
                    <div className="config-grid">
                        <div className="form-group">
                            <label className="form-label">Random Seed</label>
                            <input
                                type="number"
                                className="form-input"
                                value={config.seed}
                                onChange={(e) => handleConfigChange('seed', parseInt(e.target.value))}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Sample Size (%)</label>
                            <input
                                type="number"
                                className="form-input"
                                value={config.sampleSize}
                                onChange={(e) => handleConfigChange('sampleSize', parseInt(e.target.value))}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Validation Split (0-1)</label>
                            <input
                                type="number"
                                step="0.1"
                                className="form-input"
                                value={config.validationSize}
                                onChange={(e) => handleConfigChange('validationSize', parseFloat(e.target.value))}
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
                            </select>
                        </div>
                    </div>
                </div>

                {/* Training Settings */}
                <div className="section">
                    <div className="section-title">
                        <Activity size={20} />
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
                            <label className="form-label">Save Checkpoints</label>
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.saveCheckpoints}
                                    onChange={(e) => handleConfigChange('saveCheckpoints', e.target.checked)}
                                />
                                <div className="checkbox-visual">
                                    {config.saveCheckpoints && <div style={{ width: 8, height: 8, background: '#fff', borderRadius: 2 }} />}
                                </div>
                                <span>Enable</span>
                            </label>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Run Tensorboard</label>
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.runTensorboard}
                                    onChange={(e) => handleConfigChange('runTensorboard', e.target.checked)}
                                />
                                <div className="checkbox-visual">
                                    {config.runTensorboard && <div style={{ width: 8, height: 8, background: '#fff', borderRadius: 2 }} />}
                                </div>
                                <span>Enable</span>
                            </label>
                        </div>
                    </div>
                </div>

                {/* Model Settings */}
                <div className="section">
                    <div className="section-title">
                        <Cpu size={20} />
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
                            <label className="form-label">Embedding Dims</label>
                            <input
                                type="number"
                                className="form-input"
                                value={config.embeddingDims}
                                onChange={(e) => handleConfigChange('embeddingDims', parseInt(e.target.value))}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Mixed Precision</label>
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.mixedPrecision}
                                    onChange={(e) => handleConfigChange('mixedPrecision', e.target.checked)}
                                />
                                <div className="checkbox-visual">
                                    {config.mixedPrecision && <div style={{ width: 8, height: 8, background: '#fff', borderRadius: 2 }} />}
                                </div>
                                <span>Enable</span>
                            </label>
                        </div>
                    </div>
                </div>
            </div>

            {/* Actions */}
            <div className="actions-bar">
                <button className="btn btn-secondary">
                    <Save size={18} />
                    Save Config
                </button>
                <button className="btn btn-primary">
                    <Play size={18} />
                    Start Training
                </button>
            </div>
        </div>
    );
}
