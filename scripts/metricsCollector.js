/**
 * MetricsCollector - minimal stub for severityClassifier.js integration.
 * Provides no-op MetricTimer for environments where metrics are not needed.
 * Replace with a real implementation (e.g. Prometheus, DataDog) for production.
 */

class NoOpTimer {
  constructor() {}

  recordItems(count, errorCount) {
    // no-op
  }

  addMetadata() {
    // no-op
  }

  async end() {
    // no-op
    return { durationMs: 0, itemCount: 0, errorCount: 0 };
  }
}

class MetricTimer {
  /**
   * @param {string} name
   * @param {string} type
   * @param {string} category
   */
  constructor(name, type, category) {
    this.name = name;
    this.type = type;
    this.category = category;
    this.startTime = Date.now();
    this.metadata = {};
    this.itemCount = 0;
    this.errorCount = 0;
  }

  recordItems(count, errorCount = 0) {
    this.itemCount = count;
    this.errorCount = errorCount;
  }

  addMetadata(key, value) {
    this.metadata[key] = value;
  }

  async end() {
    const durationMs = Date.now() - this.startTime;
    return {
      name: this.name,
      type: this.type,
      category: this.category,
      durationMs,
      itemCount: this.itemCount,
      errorCount: this.errorCount,
      metadata: this.metadata,
    };
  }
}

module.exports = { MetricTimer, NoOpTimer };