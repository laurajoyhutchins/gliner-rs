/**
 * One authority for a live model: publish only a completely initialized
 * replacement. An interrupted or failed load never frees the current model.
 */
export interface DisposableModel {
  model: { free(): void };
}

export class ModelLifecycle<T extends DisposableModel> {
  private generation = 0;
  private pending: AbortController | null = null;

  constructor(
    private readonly current: () => T | null,
    private readonly publish: (model: T | null) => void,
  ) {}

  async replace(load: (signal: AbortSignal) => Promise<T>, committed?: (model: T) => void): Promise<T> {
    this.pending?.abort();
    const generation = ++this.generation;
    const controller = new AbortController();
    this.pending = controller;
    let loaded: T | null = null;
    try {
      loaded = await load(controller.signal);
      if (controller.signal.aborted || generation !== this.generation) {
        loaded.model.free();
        loaded = null;
        throw new DOMException("Model selection was superseded", "AbortError");
      }

      const previous = this.current();
      this.publish(loaded);
      committed?.(loaded);
      if (previous && previous.model !== loaded.model) previous.model.free();
      return loaded;
    } finally {
      if (generation === this.generation) this.pending = null;
    }
  }

  /** Invalidate any pending load without replacing the existing model. */
  cancelPending(): void {
    this.generation++;
    this.pending?.abort();
    this.pending = null;
  }

  unload(): void {
    this.cancelPending();
    const previous = this.current();
    this.publish(null);
    previous?.model.free();
  }
}
