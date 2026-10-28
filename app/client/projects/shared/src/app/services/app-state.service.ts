import { Injectable, signal } from '@angular/core';
import { InferencePageState, InferenceStudy } from '../types';

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
  readonly inference = signal<InferencePageState>(freshInferenceState());
  // Studies of the models that are not selected, keyed by model ref; never touched by model memory release
  readonly inferenceStudies = signal<Record<string, InferenceStudy>>({});

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
}
