import { Injectable, signal } from '@angular/core';
import { DatasetPageState, DatasetProcessingConfig, InferencePageState, InferenceStudy, TrainingConfig, TrainingDashboardState, TrainingPageState } from '../types';

const defaultDatasetConfig: DatasetProcessingConfig = { datasetName: '', sampleSize: 1, validationSize: 0.2, maxReportSize: 200, tokenizer: 'distilbert-base-uncased', evalSampleSize: 0.1, imgStats: false, textStats: false, pixDist: false };
const defaultTrainingConfig: TrainingConfig = { architecturePreset: 'xreport-default', useImgAugment: false, epochs: 100, batchSize: 32, gradientAccumulationSteps: 1, saveCheckpoints: true, useScheduler: false, targetLR: 0.001, warmupSteps: 1000, realTimePlot: true, dataloaderWorkers: 0, prefetchFactor: 1, pinMemory: true, persistentWorkers: false, useMixedPrecision: false, useGpu: true, gpuId: 0 };
const defaultDashboard: TrainingDashboardState = { isTraining: false, currentEpoch: 0, totalEpochs: 0, loss: 0, valLoss: 0, accuracy: 0, valAccuracy: 0, progressPercent: 0, elapsedSeconds: 0, workerPhase: 'idle', workerPhaseStatus: 'idle', workerPhaseElapsedSeconds: 0, workerElapsedSeconds: 0, chartData: [], availableMetrics: [], epochBoundaries: [], logEntries: [] };

function freshDatasetState(): DatasetPageState {
  return { config: { ...defaultDatasetConfig }, imageFolderPath: '', imageFolderName: '', imageValidation: null, datasetFile: null, datasetUpload: null, loadResult: null, isLoading: false, uploadError: null, isProcessing: false, processingResult: null, dbStatus: null, datasetNames: null, selectedDatasets: [], isValidating: false, validationError: null };
}
function freshTrainingState(): TrainingPageState {
  return { config: { ...defaultTrainingConfig }, newSessionExpanded: true, resumeSessionExpanded: false, selectedCheckpoint: '', additionalEpochs: 50, dashboardState: { ...defaultDashboard } };
}
function freshInferenceStudy(): InferenceStudy {
  return { images: [], currentIndex: 0, generatedReport: '', runScope: 'selected', runIndices: [], studyReport: false, requestId: null, generationProfile: 'deterministic', clinicalContext: '', reports: {}, drafts: {}, baselineDrafts: {}, persistedHistory: null };
}
function freshInferenceState(): InferencePageState {
  return { ...freshInferenceStudy(), isGenerating: false, activeJobId: null, isCopied: false, selectedModelRef: '', modelAvailability: [], isLoadingModels: false, streamingTokens: '', currentStreamingIndex: -1 };
}
function studyOf(state: InferencePageState): InferenceStudy {
  const { images, currentIndex, generatedReport, runScope, runIndices, studyReport, requestId, generationProfile, clinicalContext, reports, drafts, baselineDrafts, persistedHistory } = state;
  return { images, currentIndex, generatedReport, runScope, runIndices, studyReport, requestId, generationProfile, clinicalContext, reports, drafts, baselineDrafts, persistedHistory };
}

@Injectable({ providedIn: 'root' })
export class AppStateService {
  readonly dataset = signal<DatasetPageState>(freshDatasetState());
  readonly training = signal<TrainingPageState>(freshTrainingState());
  readonly inference = signal<InferencePageState>(freshInferenceState());
  // Studies of the models that are not selected, keyed by model ref; never touched by model memory release
  readonly inferenceStudies = signal<Record<string, InferenceStudy>>({});

  updateDataset(updater: (state: DatasetPageState) => DatasetPageState) { this.dataset.update(updater); }
  updateTraining(updater: (state: TrainingPageState) => TrainingPageState) { this.training.update(updater); }
  updateInference(updater: (state: InferencePageState) => InferencePageState) { this.inference.update(updater); }
  /** Keeps the current model's study aside and restores the target model's study, or starts an empty one. */
  switchInferenceModel(modelRef: string) {
    const state = this.inference();
    if (modelRef === state.selectedModelRef) return;
    const restored = this.inferenceStudies()[modelRef] ?? freshInferenceStudy();
    this.inferenceStudies.update((studies) => {
      const rest = Object.fromEntries(Object.entries(studies).filter(([ref]) => ref !== modelRef));
      return state.selectedModelRef ? { ...rest, [state.selectedModelRef]: studyOf(state) } : rest;
    });
    this.inference.set({ ...state, ...restored, selectedModelRef: modelRef, isCopied: false });
  }
  /** Resets the selected model's study, keeping its generation profile and run scope. */
  clearInferenceStudy() { this.updateInference((state) => ({ ...state, ...freshInferenceStudy(), generationProfile: state.generationProfile, runScope: state.runScope, isCopied: false })); }
  /** Drops kept studies of models that left the catalogue. */
  retainInferenceStudies(modelRefs: ReadonlySet<string>) { this.inferenceStudies.update((studies) => Object.fromEntries(Object.entries(studies).filter(([modelRef]) => modelRefs.has(modelRef)))); }
  updateDatasetConfig<K extends keyof DatasetProcessingConfig>(key: K, value: DatasetProcessingConfig[K]) { this.updateDataset((state) => ({ ...state, config: { ...state.config, [key]: value } })); }
  updateTrainingConfig<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) { this.updateTraining((state) => ({ ...state, config: { ...state.config, [key]: value } })); }
  updateDashboard(updater: Partial<TrainingDashboardState> | ((state: TrainingDashboardState) => TrainingDashboardState)) { this.updateTraining((state) => ({ ...state, dashboardState: typeof updater === 'function' ? updater(state.dashboardState) : { ...state.dashboardState, ...updater } })); }
}
