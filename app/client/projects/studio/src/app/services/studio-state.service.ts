import { Injectable, signal } from '@angular/core';
import { DatasetPageState, DatasetProcessingConfig, TrainingConfig, TrainingDashboardState, TrainingPageState } from '../types/studioState';

const defaultDatasetConfig: DatasetProcessingConfig = { datasetName: '', sampleSize: 1, validationSize: 0.2, maxReportSize: 200, tokenizer: 'distilbert-base-uncased', evalSampleSize: 0.1, imgStats: false, textStats: false, pixDist: false };
const defaultTrainingConfig: TrainingConfig = { architecturePreset: 'xreport-default', useImgAugment: false, epochs: 100, batchSize: 32, gradientAccumulationSteps: 1, saveCheckpoints: true, useScheduler: false, targetLR: 0.001, warmupSteps: 1000, realTimePlot: true, dataloaderWorkers: 0, prefetchFactor: 1, pinMemory: true, persistentWorkers: false, useMixedPrecision: false, useGpu: true, gpuId: 0 };
const defaultDashboard: TrainingDashboardState = { isTraining: false, currentEpoch: 0, totalEpochs: 0, loss: 0, valLoss: 0, accuracy: 0, valAccuracy: 0, progressPercent: 0, elapsedSeconds: 0, workerPhase: 'idle', workerPhaseStatus: 'idle', workerPhaseElapsedSeconds: 0, workerElapsedSeconds: 0, chartData: [], availableMetrics: [], epochBoundaries: [], logEntries: [] };

function freshDatasetState(): DatasetPageState {
  return { config: { ...defaultDatasetConfig }, imageFolderPath: '', imageFolderName: '', imageValidation: null, datasetFile: null, datasetUpload: null, loadResult: null, isLoading: false, uploadError: null, isProcessing: false, processingResult: null, dbStatus: null, datasetNames: null, selectedDatasets: [], isValidating: false, validationError: null };
}
function freshTrainingState(): TrainingPageState {
  return { config: { ...defaultTrainingConfig }, newSessionExpanded: true, resumeSessionExpanded: false, selectedCheckpoint: '', additionalEpochs: 50, dashboardState: { ...defaultDashboard } };
}
/** Dataset and training page state, kept across navigation inside XREPORT Studio. */
@Injectable({ providedIn: 'root' })
export class StudioStateService {
  readonly dataset = signal<DatasetPageState>(freshDatasetState());
  readonly training = signal<TrainingPageState>(freshTrainingState());

  updateDataset(updater: (state: DatasetPageState) => DatasetPageState) { this.dataset.update(updater); }
  updateTraining(updater: (state: TrainingPageState) => TrainingPageState) { this.training.update(updater); }
  updateDatasetConfig<K extends keyof DatasetProcessingConfig>(key: K, value: DatasetProcessingConfig[K]) { this.updateDataset((state) => ({ ...state, config: { ...state.config, [key]: value } })); }
  updateTrainingConfig<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) { this.updateTraining((state) => ({ ...state, config: { ...state.config, [key]: value } })); }
  updateDashboard(updater: Partial<TrainingDashboardState> | ((state: TrainingDashboardState) => TrainingDashboardState)) { this.updateTraining((state) => ({ ...state, dashboardState: typeof updater === 'function' ? updater(state.dashboardState) : { ...state.dashboardState, ...updater } })); }
}
